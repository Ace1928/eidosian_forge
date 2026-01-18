import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def check_exceptions(self, pyfunc, *args, **kwargs):
    """
        For a given kernel:

        The expected result is computed from a pyStencil version of the
        stencil.

        The following results are then computed:
        * from a pure @stencil decoration of the kernel.
        * from the njit of a trivial wrapper function around the pure @stencil
          decorated function.
        * from the njit(parallel=True) of a trivial wrapper function around
           the pure @stencil decorated function.

        The results are then compared.
        """
    options = kwargs.get('options', dict())
    expected_exception = kwargs.get('expected_exception')
    should_fail = []
    should_not_fail = []

    @contextmanager
    def errorhandler(exty=None, usecase=None):
        try:
            yield
        except Exception as e:
            if exty is not None:
                lexty = exty if hasattr(exty, '__iter__') else [exty]
                found = False
                for ex in lexty:
                    found |= isinstance(e, ex)
                if not found:
                    raise
            else:
                should_not_fail.append((usecase, '%s: %s' % (type(e), str(e))))
        else:
            if exty is not None:
                should_fail.append(usecase)
    if isinstance(expected_exception, dict):
        stencil_ex = expected_exception['stencil']
        njit_ex = expected_exception['njit']
        parfor_ex = expected_exception['parfor']
    else:
        stencil_ex = expected_exception
        njit_ex = expected_exception
        parfor_ex = expected_exception
    stencil_args = {'func_or_mode': pyfunc}
    stencil_args.update(options)
    with errorhandler(stencil_ex, '@stencil'):
        stencil_func_impl = stencil(**stencil_args)
        stencil_func_impl(*args)
    if len(args) == 1:

        def wrap_stencil(arg0):
            return stencil_func_impl(arg0)
    elif len(args) == 2:

        def wrap_stencil(arg0, arg1):
            return stencil_func_impl(arg0, arg1)
    elif len(args) == 3:

        def wrap_stencil(arg0, arg1, arg2):
            return stencil_func_impl(arg0, arg1, arg2)
    else:
        raise ValueError('Up to 3 arguments can be provided, found %s' % len(args))
    sig = tuple([numba.typeof(x) for x in args])
    with errorhandler(njit_ex, 'njit'):
        wrapped_cfunc = self.compile_njit(wrap_stencil, sig)
        wrapped_cfunc.entry_point(*args)
    with errorhandler(parfor_ex, 'parfors'):
        wrapped_cpfunc = self.compile_parallel(wrap_stencil, sig)
        wrapped_cpfunc.entry_point(*args)
    if should_fail:
        msg = ['%s' % x for x in should_fail]
        raise RuntimeError('The following implementations should have raised an exception but did not:\n%s' % msg)
    if should_not_fail:
        impls = ['%s' % x[0] for x in should_not_fail]
        errs = ''.join(['%s: Message: %s\n\n' % x for x in should_not_fail])
        str1 = 'The following implementations should not have raised an exception but did:\n%s\n' % impls
        str2 = 'Errors were:\n\n%s' % errs
        raise RuntimeError(str1 + str2)