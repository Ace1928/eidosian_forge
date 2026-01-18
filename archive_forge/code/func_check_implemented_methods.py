from xarray import apply_ufunc
from ..stats import wrap_xarray_ufunc as _wrap_xarray_ufunc
def check_implemented_methods(self, methods):
    """Check that all methods listed are implemented.

        Not all functions that require refitting need to have all the methods implemented in
        order to work properly. This function shoulg be used before using the SamplingWrapper and
        its subclasses to get informative error messages.

        Parameters
        ----------
        methods: list
            Check all elements in methods are implemented.

        Returns
        -------
            List with all non implemented methods
        """
    supported_methods_1arg = ('sel_observations', 'sample', 'get_inference_data')
    supported_methods_2args = ('log_likelihood__i',)
    supported_methods = [*supported_methods_1arg, *supported_methods_2args]
    bad_methods = [method for method in methods if method not in supported_methods]
    if bad_methods:
        raise ValueError(f'Not all method(s) in {bad_methods} supported. Supported methods in SamplingWrapper subclasses are:{supported_methods}')
    not_implemented = []
    for method in methods:
        if method in supported_methods_1arg:
            if self._check_method_is_implemented(method, 1):
                continue
            not_implemented.append(method)
        elif method in supported_methods_2args:
            if self._check_method_is_implemented(method, 1, 1):
                continue
            not_implemented.append(method)
    return not_implemented