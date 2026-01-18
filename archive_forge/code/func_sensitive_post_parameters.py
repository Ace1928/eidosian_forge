import inspect
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpRequest
def sensitive_post_parameters(*parameters):
    """
    Indicate which POST parameters used in the decorated view are sensitive,
    so that those parameters can later be treated in a special way, for example
    by hiding them when logging unhandled exceptions.

    Accept two forms:

    * with specified parameters:

        @sensitive_post_parameters('password', 'credit_card')
        def my_view(request):
            pw = request.POST['password']
            cc = request.POST['credit_card']
            ...

    * without any specified parameters, in which case consider all
      variables are sensitive:

        @sensitive_post_parameters()
        def my_view(request)
            ...
    """
    if len(parameters) == 1 and callable(parameters[0]):
        raise TypeError('sensitive_post_parameters() must be called to use it as a decorator, e.g., use @sensitive_post_parameters(), not @sensitive_post_parameters.')

    def decorator(view):
        if iscoroutinefunction(view):

            @wraps(view)
            async def sensitive_post_parameters_wrapper(request, *args, **kwargs):
                if not isinstance(request, HttpRequest):
                    raise TypeError("sensitive_post_parameters didn't receive an HttpRequest object. If you are decorating a classmethod, make sure to use @method_decorator.")
                if parameters:
                    request.sensitive_post_parameters = parameters
                else:
                    request.sensitive_post_parameters = '__ALL__'
                return await view(request, *args, **kwargs)
        else:

            @wraps(view)
            def sensitive_post_parameters_wrapper(request, *args, **kwargs):
                if not isinstance(request, HttpRequest):
                    raise TypeError("sensitive_post_parameters didn't receive an HttpRequest object. If you are decorating a classmethod, make sure to use @method_decorator.")
                if parameters:
                    request.sensitive_post_parameters = parameters
                else:
                    request.sensitive_post_parameters = '__ALL__'
                return view(request, *args, **kwargs)
        return sensitive_post_parameters_wrapper
    return decorator