import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def inspect_function(function, target=None):
    """Return information about the support of a function.

    Returns
    -------
    info : dict
        Defined keys:
        - "numba_type": str or None
            The numba type object of the function if supported.
        - "explained": str
            A textual description of the support.
        - "source_infos": dict
            A dictionary containing the source location of each definition.
    """
    target = target or cpu_target
    tyct = target.typing_context
    tyct.refresh()
    target.target_context.refresh()
    info = {}
    source_infos = {}
    try:
        nbty = tyct.resolve_value_type(function)
    except ValueError:
        nbty = None
        explained = 'not supported'
    else:
        explained = tyct.explain_function_type(nbty)
        for temp in nbty.templates:
            try:
                source_infos[temp] = temp.get_source_info()
            except AttributeError:
                source_infos[temp] = None
    info['numba_type'] = nbty
    info['explained'] = explained
    info['source_infos'] = source_infos
    return info