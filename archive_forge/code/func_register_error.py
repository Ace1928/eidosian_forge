import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def register_error(func_name=None, cls=None):
    """Register an error class so it can be recognized by the ffi error handler.

    Parameters
    ----------
    func_name : str or function or class
        The name of the error function.

    cls : function
        The function to create the class

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    .. code-block:: python

      @mxnet.error.register_error
      class MyError(RuntimeError):
          pass

      err_inst = mxnet.error.create_ffi_error("MyError: xyz")
      assert isinstance(err_inst, MyError)
    """
    if callable(func_name):
        cls = func_name
        func_name = cls.__name__

    def register(mycls):
        """internal register function"""
        err_name = func_name if isinstance(func_name, str) else mycls.__name__
        error_types[err_name] = mycls
        return mycls
    if cls is None:
        return register
    return register(cls)