from types import FunctionType
import sys
def getFrameInfo(frame):
    """Return (kind,module,locals,globals) for a frame

    'kind' is one of "exec", "module", "class", "function call", or "unknown".
    """
    f_locals = frame.f_locals
    f_globals = frame.f_globals
    sameNamespace = f_locals is f_globals
    hasModule = '__module__' in f_locals
    hasName = '__name__' in f_globals
    sameName = hasModule and hasName
    sameName = sameName and f_globals['__name__'] == f_locals['__module__']
    module = hasName and sys.modules.get(f_globals['__name__']) or None
    namespaceIsModule = module and module.__dict__ is f_globals
    if not namespaceIsModule:
        kind = 'exec'
    elif sameNamespace and (not hasModule):
        kind = 'module'
    elif sameName and (not sameNamespace):
        kind = 'class'
    elif not sameNamespace:
        kind = 'function call'
    else:
        kind = 'unknown'
    return (kind, module, f_locals, f_globals)