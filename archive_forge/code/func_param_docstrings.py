import re
import itertools
import textwrap
import uuid
import param
from param.display import register_display_accessor
from param._utils import async_executor
def param_docstrings(self, info, max_col_len=100, only_changed=False):
    """
        Build a string to that presents all of the parameter
        docstrings in a clean format (alternating red and blue for
        readability).
        """
    params, val_dict, changed = info
    contents = []
    displayed_params = []
    for name in self.sort_by_precedence(params):
        if only_changed and (not name in changed):
            continue
        displayed_params.append((name, params[name]))
    right_shift = max((len(name) for name, _ in displayed_params)) + 2
    for i, (name, p) in enumerate(displayed_params):
        heading = '%s: ' % name
        unindented = textwrap.dedent('< No docstring available >' if p.doc is None else p.doc)
        if WARN_MISFORMATTED_DOCSTRINGS and (not unindented.startswith('\n')) and (len(unindented.splitlines()) > 1):
            param.main.warning('Multi-line docstring for %r is incorrectly formatted  (should start with newline)', name)
        while unindented.startswith('\n'):
            unindented = unindented[1:]
        lines = unindented.splitlines()
        if len(lines) > 1:
            tail = [f'{' ' * right_shift}{line}' for line in lines[1:]]
            all_lines = [heading.ljust(right_shift) + lines[0]] + tail
        elif len(lines) == 1:
            all_lines = [heading.ljust(right_shift) + lines[0]]
        else:
            all_lines = []
        if i % 2:
            contents.extend([red % el for el in all_lines])
        else:
            contents.extend([blue % el for el in all_lines])
    return '\n'.join(contents)