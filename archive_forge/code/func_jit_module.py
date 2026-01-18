import sys
import warnings
import inspect
import logging
from numba.core.errors import DeprecationError, NumbaDeprecationWarning
from numba.stencils.stencil import stencil
from numba.core import config, extending, sigutils, registry
def jit_module(**kwargs):
    """ Automatically ``jit``-wraps functions defined in a Python module

    Note that ``jit_module`` should only be called at the end of the module to
    be jitted. In addition, only functions which are defined in the module
    ``jit_module`` is called from are considered for automatic jit-wrapping.
    See the Numba documentation for more information about what can/cannot be
    jitted.

    :param kwargs: Keyword arguments to pass to ``jit`` such as ``nopython``
                   or ``error_model``.

    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    for name, obj in module.__dict__.items():
        if inspect.isfunction(obj) and inspect.getmodule(obj) == module:
            _logger.debug('Auto decorating function {} from module {} with jit and options: {}'.format(obj, module.__name__, kwargs))
            module.__dict__[name] = jit(obj, **kwargs)