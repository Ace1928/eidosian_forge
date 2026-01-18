import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@classmethod
def options_info(cls, plot_class, ansi=False, pattern=None):
    if plot_class.style_opts:
        backend_name = plot_class.backend
        style_info = f"\n(Consult {backend_name}'s documentation for more information.)"
        style_keywords = f'\t{', '.join(plot_class.style_opts)}'
        style_msg = f'{style_keywords}\n{style_info}'
    else:
        style_msg = '\t<No style options available>'
    param_info = cls.get_parameter_info(plot_class, ansi=ansi, pattern=pattern)
    lines = [cls.heading('Style Options', ansi=ansi, char='-'), '', style_msg, '', cls.heading('Plot Options', ansi=ansi, char='-'), '']
    if param_info is not None:
        lines += ['The plot options are the parameters of the plotting class:\n', param_info]
    elif pattern is not None:
        lines += [f'No {plot_class.__name__!r} parameters found matching specified pattern {pattern!r}.']
    else:
        lines += [f'No {plot_class.__name__!r} parameters found.']
    return '\n'.join(lines)