import linecache
import re
def default_should_trace_hook(frame, absolute_filename):
    """
    Return True if this frame should be traced, False if tracing should be blocked.
    """
    ignored_lines = _filename_to_ignored_lines.get(absolute_filename)
    if ignored_lines is None:
        ignored_lines = {}
        lines = linecache.getlines(absolute_filename)
        for i_line, line in enumerate(lines):
            j = line.find('#')
            if j >= 0:
                comment = line[j:]
                if DONT_TRACE_TAG in comment:
                    ignored_lines[i_line] = 1
                    k = i_line - 1
                    while k >= 0:
                        if RE_DECORATOR.match(lines[k]):
                            ignored_lines[k] = 1
                            k -= 1
                        else:
                            break
                    k = i_line + 1
                    while k <= len(lines):
                        if RE_DECORATOR.match(lines[k]):
                            ignored_lines[k] = 1
                            k += 1
                        else:
                            break
        _filename_to_ignored_lines[absolute_filename] = ignored_lines
    func_line = frame.f_code.co_firstlineno - 1
    return not (func_line - 1 in ignored_lines or func_line in ignored_lines)