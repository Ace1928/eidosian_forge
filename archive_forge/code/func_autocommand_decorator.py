from .autoparse import autoparse
from .automain import automain
def autocommand_decorator(func):
    if loop is not None or forever or pass_loop:
        func = autoasync(func, loop=None if loop is True else loop, pass_loop=pass_loop, forever=forever)
    func = autoparse(func, description=description, epilog=epilog, add_nos=add_nos, parser=parser)
    func = automain(module)(func)
    return func