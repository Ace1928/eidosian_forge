from asyncio import get_event_loop, iscoroutine
from functools import wraps
from inspect import signature
@wraps(coro)
def autoasync_wrapper(*args, **kwargs):
    local_loop = get_event_loop() if loop is None else loop
    if pass_loop:
        bound_args = old_sig.bind_partial()
        bound_args.arguments.update(loop=local_loop, **new_sig.bind(*args, **kwargs).arguments)
        args, kwargs = (bound_args.args, bound_args.kwargs)
    if forever:
        local_loop.create_task(_run_forever_coro(coro, args, kwargs, local_loop))
        local_loop.run_forever()
    else:
        return local_loop.run_until_complete(coro(*args, **kwargs))