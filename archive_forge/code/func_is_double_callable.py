import inspect
from .sync import iscoroutinefunction
def is_double_callable(application):
    """
    Tests to see if an application is a legacy-style (double-callable) application.
    """
    if getattr(application, '_asgi_single_callable', False):
        return False
    if getattr(application, '_asgi_double_callable', False):
        return True
    if inspect.isclass(application):
        return True
    if hasattr(application, '__call__'):
        if iscoroutinefunction(application.__call__):
            return False
    return not iscoroutinefunction(application)