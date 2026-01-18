import unittest
from contextlib import contextmanager
from llvmlite import ir
from numba.core import types, typing, callconv, cpu, cgutils
from numba.core.registry import cpu_target
def test_closures(self):
    """
        Caching must not mix up closures reusing the same code object.
        """

    def make_closure(x, y):

        def f(z):
            return y + z
        return f
    with self._context_builder_sig_args() as (context, builder, sig, args):
        clo11 = make_closure(1, 1)
        clo12 = make_closure(1, 2)
        clo22 = make_closure(2, 2)
        initial_cache_size = len(context.cached_internal_func)
        res1 = context.compile_internal(builder, clo11, sig, args)
        self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
        res2 = context.compile_internal(builder, clo12, sig, args)
        self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))
        res3 = context.compile_internal(builder, clo22, sig, args)
        self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))