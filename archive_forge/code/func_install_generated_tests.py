from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
@classmethod
def install_generated_tests(cls):
    reduction_funcs = [array_sum, array_sum_global, array_prod, array_prod_global, array_mean, array_mean_global, array_var, array_var_global, array_std, array_std_global, array_all, array_all_global, array_any, array_any_global, array_min, array_min_global, array_amax, array_amin, array_max, array_max_global, array_nanmax, array_nanmin, array_nansum]
    reduction_funcs_rspace = [array_argmin, array_argmin_global, array_argmax, array_argmax_global]
    reduction_funcs += [array_nanmean, array_nanstd, array_nanvar]
    reduction_funcs += [array_nanprod]
    dtypes_to_test = [np.int32, np.float32, np.bool_, np.complex64]

    def install_tests(dtypes, funcs):
        for dt in dtypes:
            test_arrays = full_test_arrays(dt)
            for red_func, test_array in product(funcs, test_arrays):
                test_name = 'test_{0}_{1}_{2}d'
                test_name = test_name.format(red_func.__name__, test_array.dtype.name, test_array.ndim)

                def new_test_function(self, redFunc=red_func, testArray=test_array, testName=test_name):
                    ulps = 1
                    if 'prod' in red_func.__name__ and np.iscomplexobj(testArray):
                        ulps = 3
                    npr, nbr = run_comparative(redFunc, testArray)
                    self.assertPreciseEqual(npr, nbr, msg=testName, prec='single', ulps=ulps)
                setattr(cls, test_name, new_test_function)
    install_tests(dtypes_to_test[:-1], reduction_funcs_rspace)
    install_tests(dtypes_to_test, reduction_funcs)