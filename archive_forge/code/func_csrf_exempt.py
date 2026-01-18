from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.middleware.csrf import CsrfViewMiddleware, get_token
from django.utils.decorators import decorator_from_middleware
def csrf_exempt(view_func):
    """Mark a view function as being exempt from the CSRF view protection."""
    if iscoroutinefunction(view_func):

        async def _view_wrapper(request, *args, **kwargs):
            return await view_func(request, *args, **kwargs)
    else:

        def _view_wrapper(request, *args, **kwargs):
            return view_func(request, *args, **kwargs)
    _view_wrapper.csrf_exempt = True
    return wraps(view_func)(_view_wrapper)