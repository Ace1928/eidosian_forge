from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.middleware.cache import CacheMiddleware
from django.utils.cache import add_never_cache_headers, patch_cache_control
from django.utils.decorators import decorator_from_middleware_with_args
def never_cache(view_func):
    """
    Decorator that adds headers to a response so that it will never be cached.
    """
    if iscoroutinefunction(view_func):

        async def _view_wrapper(request, *args, **kwargs):
            _check_request(request, 'never_cache')
            response = await view_func(request, *args, **kwargs)
            add_never_cache_headers(response)
            return response
    else:

        def _view_wrapper(request, *args, **kwargs):
            _check_request(request, 'never_cache')
            response = view_func(request, *args, **kwargs)
            add_never_cache_headers(response)
            return response
    return wraps(view_func)(_view_wrapper)