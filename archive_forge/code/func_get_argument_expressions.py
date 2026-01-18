import re
from mako import exceptions
from mako import pyparser
def get_argument_expressions(self, as_call=False):
    """Return the argument declarations of this FunctionDecl as a printable
        list.

        By default the return value is appropriate for writing in a ``def``;
        set `as_call` to true to build arguments to be passed to the function
        instead (assuming locals with the same names as the arguments exist).
        """
    namedecls = []
    argnames = self.argnames[::-1]
    kwargnames = self.kwargnames[::-1]
    defaults = self.defaults[::-1]
    kwdefaults = self.kwdefaults[::-1]
    if self.kwargs:
        namedecls.append('**' + kwargnames.pop(0))
    for name in kwargnames:
        if as_call:
            namedecls.append('%s=%s' % (name, name))
        elif kwdefaults:
            default = kwdefaults.pop(0)
            if default is None:
                namedecls.append(name)
            else:
                namedecls.append('%s=%s' % (name, pyparser.ExpressionGenerator(default).value()))
        else:
            namedecls.append(name)
    if self.varargs:
        namedecls.append('*' + argnames.pop(0))
    for name in argnames:
        if as_call or not defaults:
            namedecls.append(name)
        else:
            default = defaults.pop(0)
            namedecls.append('%s=%s' % (name, pyparser.ExpressionGenerator(default).value()))
    namedecls.reverse()
    return namedecls