from __future__ import (absolute_import, division, print_function)
from jinja2.nativetypes import NativeTemplate
def new_context(self, vars=None, shared=False, locals=None):
    if vars is None:
        vars = dict(self.globals or ())
    if isinstance(vars, dict):
        vars = vars.copy()
        if locals is not None:
            vars.update(locals)
    else:
        vars = vars.add_locals(locals)
    return self.environment.context_class(self.environment, vars, self.name, self.blocks)