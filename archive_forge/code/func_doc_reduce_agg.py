from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
def doc_reduce_agg(method, refer_to, params=None, extra_params=None):
    """
    Build decorator which adds docstring for the reduce method.

    Parameters
    ----------
    method : str
        The result of the method.
    refer_to : str
        Method name in ``modin.pandas.DataFrame`` module to refer to for
        more information about parameters and output format.
    params : str, optional
        Method parameters in the NumPy docstyle format to substitute
        to the docstring template.
    extra_params : sequence of str, optional
        Method parameter names to append to the docstring template. Parameter
        type and description will be grabbed from ``extra_params_map`` (Please
        refer to the source code of this function to explore the map).

    Returns
    -------
    callable
    """
    template = '\n        Get the {method} for each column or row.\n        {params}\n        Returns\n        -------\n        BaseQueryCompiler\n            One-column QueryCompiler with index labels of the specified axis,\n            where each row contains the {method} for the corresponding\n            row or column.\n        '
    if params is None:
        params = '\n        axis : {{0, 1}}\n        numeric_only : bool, optional'
    extra_params_map = {'skipna': '\n        skipna : bool, default: True', 'min_count': '\n        min_count : int', 'ddof': '\n        ddof : int', '*args': '\n        *args : iterable\n            Serves the compatibility purpose. Does not affect the result.', '**kwargs': '\n        **kwargs : dict\n            Serves the compatibility purpose. Does not affect the result.'}
    params += ''.join([align_indents(source=params, target=extra_params_map.get(param, f'\n{param} : object')) for param in extra_params or []])
    return doc_qc_method(template, params=params, method=method, refer_to=f'DataFrame.{refer_to}')