import functools
import re
import types
def replace_return_docstrings(output_type=None, config_class=None):

    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split('\n')
        i = 0
        while i < len(lines) and re.search('^\\s*Returns?:\\s*$', lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = '\n'.join(lines)
        else:
            raise ValueError(f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{func_doc}")
        fn.__doc__ = func_doc
        return fn
    return docstring_decorator