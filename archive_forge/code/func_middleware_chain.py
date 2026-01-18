import inspect
from functools import partial
from itertools import chain
from wandb_promise import Promise
def middleware_chain(func, middlewares, wrap_in_promise):
    if not middlewares:
        return func
    if wrap_in_promise:
        middlewares = chain((func, make_it_promise), middlewares)
    else:
        middlewares = chain((func,), middlewares)
    last_func = None
    for middleware in middlewares:
        last_func = partial(middleware, last_func) if last_func else middleware
    return last_func