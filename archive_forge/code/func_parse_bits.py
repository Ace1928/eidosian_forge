from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError
def parse_bits(parser, bits, params, varargs, varkw, defaults, kwonly, kwonly_defaults, takes_context, name):
    """
    Parse bits for template tag helpers simple_tag and inclusion_tag, in
    particular by detecting syntax errors and by extracting positional and
    keyword arguments.
    """
    if takes_context:
        if params and params[0] == 'context':
            params = params[1:]
        else:
            raise TemplateSyntaxError("'%s' is decorated with takes_context=True so it must have a first argument of 'context'" % name)
    args = []
    kwargs = {}
    unhandled_params = list(params)
    unhandled_kwargs = [kwarg for kwarg in kwonly if not kwonly_defaults or kwarg not in kwonly_defaults]
    for bit in bits:
        kwarg = token_kwargs([bit], parser)
        if kwarg:
            param, value = kwarg.popitem()
            if param not in params and param not in kwonly and (varkw is None):
                raise TemplateSyntaxError("'%s' received unexpected keyword argument '%s'" % (name, param))
            elif param in kwargs:
                raise TemplateSyntaxError("'%s' received multiple values for keyword argument '%s'" % (name, param))
            else:
                kwargs[str(param)] = value
                if param in unhandled_params:
                    unhandled_params.remove(param)
                elif param in unhandled_kwargs:
                    unhandled_kwargs.remove(param)
        elif kwargs:
            raise TemplateSyntaxError("'%s' received some positional argument(s) after some keyword argument(s)" % name)
        else:
            args.append(parser.compile_filter(bit))
            try:
                unhandled_params.pop(0)
            except IndexError:
                if varargs is None:
                    raise TemplateSyntaxError("'%s' received too many positional arguments" % name)
    if defaults is not None:
        unhandled_params = unhandled_params[:-len(defaults)]
    if unhandled_params or unhandled_kwargs:
        raise TemplateSyntaxError("'%s' did not receive value(s) for the argument(s): %s" % (name, ', '.join(("'%s'" % p for p in unhandled_params + unhandled_kwargs))))
    return (args, kwargs)