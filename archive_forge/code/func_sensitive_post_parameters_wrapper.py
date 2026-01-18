import inspect
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpRequest
@wraps(view)
def sensitive_post_parameters_wrapper(request, *args, **kwargs):
    if not isinstance(request, HttpRequest):
        raise TypeError("sensitive_post_parameters didn't receive an HttpRequest object. If you are decorating a classmethod, make sure to use @method_decorator.")
    if parameters:
        request.sensitive_post_parameters = parameters
    else:
        request.sensitive_post_parameters = '__ALL__'
    return view(request, *args, **kwargs)