import re
import html
from paste.util import PySourceColor
def format_collected_data(self, exc_data):
    general_data = {}
    if self.show_extra_data:
        for name, value_list in exc_data.extra_data.items():
            if isinstance(name, tuple):
                importance, title = name
            else:
                importance, title = ('normal', name)
            for value in value_list:
                general_data[importance, name] = self.format_extra_data(importance, title, value)
    lines = []
    frames = self.filter_frames(exc_data.frames)
    for frame in frames:
        sup = frame.supplement
        if sup:
            if sup.object:
                general_data['important', 'object'] = self.format_sup_object(sup.object)
            if sup.source_url:
                general_data['important', 'source_url'] = self.format_sup_url(sup.source_url)
            if sup.line:
                lines.append(self.format_sup_line_pos(sup.line, sup.column))
            if sup.expression:
                lines.append(self.format_sup_expression(sup.expression))
            if sup.warnings:
                for warning in sup.warnings:
                    lines.append(self.format_sup_warning(warning))
            if sup.info:
                lines.extend(self.format_sup_info(sup.info))
        if frame.supplement_exception:
            lines.append('Exception in supplement:')
            lines.append(self.quote_long(frame.supplement_exception))
        if frame.traceback_info:
            lines.append(self.format_traceback_info(frame.traceback_info))
        filename = frame.filename
        if filename and self.trim_source_paths:
            for path, repl in self.trim_source_paths:
                if filename.startswith(path):
                    filename = repl + filename[len(path):]
                    break
        lines.append(self.format_source_line(filename or '?', frame))
        source = frame.get_source_line()
        long_source = frame.get_source_line(2)
        if source:
            lines.append(self.format_long_source(source, long_source))
    etype = exc_data.exception_type
    if not isinstance(etype, str):
        etype = etype.__name__
    exc_info = self.format_exception_info(etype, exc_data.exception_value)
    data_by_importance = {'important': [], 'normal': [], 'supplemental': [], 'extra': []}
    for (importance, name), value in general_data.items():
        data_by_importance[importance].append((name, value))
    for value in data_by_importance.values():
        value.sort()
    return self.format_combine(data_by_importance, lines, exc_info)