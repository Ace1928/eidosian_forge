import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def careful_sub(cheetah_template, vars, filename):
    """
    Substitutes the template with the variables, using the
    .body() method if it exists.  It assumes that the variables
    were also passed in via the searchList.
    """
    if not hasattr(cheetah_template, 'body'):
        return sub_catcher(filename, vars, str, cheetah_template)
    body = cheetah_template.body
    args, varargs, varkw, defaults = inspect.getargspec(body)
    call_vars = {}
    for arg in args:
        if arg in vars:
            call_vars[arg] = vars[arg]
    return sub_catcher(filename, vars, body, **call_vars)