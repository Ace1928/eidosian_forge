import sys
def filterwarnings(action, message='', category=Warning, module='', lineno=0, append=False):
    """Insert an entry into the list of warnings filters (at the front).

    'action' -- one of "error", "ignore", "always", "default", "module",
                or "once"
    'message' -- a regex that the warning message must match
    'category' -- a class that the warning must be a subclass of
    'module' -- a regex that the module name must match
    'lineno' -- an integer line number, 0 matches all warnings
    'append' -- if true, append to the list of filters
    """
    assert action in ('error', 'ignore', 'always', 'default', 'module', 'once'), 'invalid action: %r' % (action,)
    assert isinstance(message, str), 'message must be a string'
    assert isinstance(category, type), 'category must be a class'
    assert issubclass(category, Warning), 'category must be a Warning subclass'
    assert isinstance(module, str), 'module must be a string'
    assert isinstance(lineno, int) and lineno >= 0, 'lineno must be an int >= 0'
    if message or module:
        import re
    if message:
        message = re.compile(message, re.I)
    else:
        message = None
    if module:
        module = re.compile(module)
    else:
        module = None
    _add_filter(action, message, category, module, lineno, append=append)