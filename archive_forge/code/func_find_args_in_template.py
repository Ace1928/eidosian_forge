import sys
import os
import inspect
from . import copydir
from . import command
from paste.util.template import paste_script_template_renderer
def find_args_in_template(template):
    if isinstance(template, str):
        import Cheetah.Template
        template = Cheetah.Template.Template(file=template)
    if not hasattr(template, 'body'):
        return None
    method = template.body
    args, varargs, varkw, defaults = inspect.getargspec(method)
    defaults = list(defaults or [])
    vars = []
    while args:
        if len(args) == len(defaults):
            default = defaults.pop(0)
        else:
            default = command.NoDefault
        arg = args.pop(0)
        if arg in _skip_variables:
            continue
        vars.append(var(arg, description=None, default=default))
    return vars