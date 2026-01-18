import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def formatted_argspec(funcprops, arg_pos, columns, config):
    func = funcprops.func
    args = funcprops.argspec.args
    kwargs = funcprops.argspec.defaults
    _args = funcprops.argspec.varargs
    _kwargs = funcprops.argspec.varkwargs
    is_bound_method = funcprops.is_bound_method
    kwonly = funcprops.argspec.kwonly
    kwonly_defaults = funcprops.argspec.kwonly_defaults or dict()
    arg_color = func_for_letter(config.color_scheme['name'])
    func_color = func_for_letter(config.color_scheme['name'].swapcase())
    punctuation_color = func_for_letter(config.color_scheme['punctuation'])
    token_color = func_for_letter(config.color_scheme['token'])
    bolds = {token_color: lambda x: bold(token_color(x)), arg_color: lambda x: bold(arg_color(x))}
    s = func_color(func) + arg_color(': (')
    if is_bound_method and isinstance(arg_pos, int):
        arg_pos += 1
    for i, arg in enumerate(args):
        kw = None
        if kwargs and i >= len(args) - len(kwargs):
            kw = str(kwargs[i - (len(args) - len(kwargs))])
        color = token_color if arg_pos in (i, arg) else arg_color
        if i == arg_pos or arg == arg_pos:
            color = bolds[color]
        s += color(arg)
        if kw is not None:
            s += punctuation_color('=')
            s += token_color(kw)
        if i != len(args) - 1:
            s += punctuation_color(', ')
    if _args:
        if args:
            s += punctuation_color(', ')
        s += token_color(f'*{_args}')
    if kwonly:
        if not _args:
            if args:
                s += punctuation_color(', ')
            s += punctuation_color('*')
        marker = object()
        for arg in kwonly:
            s += punctuation_color(', ')
            color = token_color
            if arg_pos:
                color = bolds[color]
            s += color(arg)
            default = kwonly_defaults.get(arg, marker)
            if default is not marker:
                s += punctuation_color('=')
                s += token_color(repr(default))
    if _kwargs:
        if args or _args or kwonly:
            s += punctuation_color(', ')
        s += token_color(f'**{_kwargs}')
    s += punctuation_color(')')
    return linesplit(s, columns)