from __future__ import annotations
import sys
import math
def array_namespace(*xs, api_version=None, _use_compat=True):
    """
    Get the array API compatible namespace for the arrays `xs`.

    `xs` should contain one or more arrays.

    Typical usage is

        def your_function(x, y):
            xp = array_api_compat.array_namespace(x, y)
            # Now use xp as the array library namespace
            return xp.mean(x, axis=0) + 2*xp.std(y, axis=0)

    api_version should be the newest version of the spec that you need support
    for (currently the compat library wrapped APIs only support v2021.12).
    """
    namespaces = set()
    for x in xs:
        if isinstance(x, (tuple, list)):
            namespaces.add(array_namespace(*x, _use_compat=_use_compat))
        elif hasattr(x, '__array_namespace__'):
            namespaces.add(x.__array_namespace__(api_version=api_version))
        elif _is_numpy_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from .. import numpy as numpy_namespace
                namespaces.add(numpy_namespace)
            else:
                import numpy as np
                namespaces.add(np)
        elif _is_cupy_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from .. import cupy as cupy_namespace
                namespaces.add(cupy_namespace)
            else:
                import cupy as cp
                namespaces.add(cp)
        elif _is_torch_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from .. import torch as torch_namespace
                namespaces.add(torch_namespace)
            else:
                import torch
                namespaces.add(torch)
        else:
            raise TypeError('The input is not a supported array type')
    if not namespaces:
        raise TypeError('Unrecognized array input')
    if len(namespaces) != 1:
        raise TypeError(f'Multiple namespaces for array inputs: {namespaces}')
    xp, = namespaces
    return xp