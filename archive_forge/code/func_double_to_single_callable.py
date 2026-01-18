import inspect
from .sync import iscoroutinefunction
def double_to_single_callable(application):
    """
    Transforms a double-callable ASGI application into a single-callable one.
    """

    async def new_application(scope, receive, send):
        instance = application(scope)
        return await instance(receive, send)
    return new_application