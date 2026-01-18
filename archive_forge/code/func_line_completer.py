import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@classmethod
def line_completer(cls, line, completions, compositor_defs):
    sorted_keys = sorted(completions.keys())
    type_keys = [key for key in sorted_keys if '.' not in key]
    completion_key, suggestions = cls.dotted_completion(line, sorted_keys, compositor_defs)
    verbose_openers = ['style(', 'plot[', 'norm{']
    if suggestions and line.endswith('.'):
        return [f'{completion_key}.{el}' for el in suggestions]
    elif not completion_key:
        return type_keys + list(compositor_defs.keys()) + verbose_openers
    if cls._inside_delims(line, '[', ']'):
        return [kw + '=' for kw in completions[completion_key][0]]
    if cls._inside_delims(line, '{', '}'):
        return ['+axiswise', '+framewise']
    style_completions = [kw + '=' for kw in completions[completion_key][1]]
    if cls._inside_delims(line, '(', ')'):
        return style_completions
    return type_keys + list(compositor_defs.keys()) + verbose_openers