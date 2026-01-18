import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def safer_globals(g=None):
    if g is None:
        g = sys._getframe(1).f_globals.copy()
    for name in ('__annotations__', '__doc__', '__loader__', '__name__', '__package__', '__spec__'):
        if name in g:
            del g[name]
        g['__builtins__'] = __rl_safe_builtins__.copy()
    return g