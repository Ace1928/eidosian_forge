import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def cell_set(cell, value):
    """Set the value of a closure cell.

    The point of this function is to set the cell_contents attribute of a cell
    after its creation. This operation is necessary in case the cell contains a
    reference to the function the cell belongs to, as when calling the
    function's constructor
    ``f = types.FunctionType(code, globals, name, argdefs, closure)``,
    closure will not be able to contain the yet-to-be-created f.

    In Python3.7, cell_contents is writeable, so setting the contents of a cell
    can be done simply using
    >>> cell.cell_contents = value

    In earlier Python3 versions, the cell_contents attribute of a cell is read
    only, but this limitation can be worked around by leveraging the Python 3
    ``nonlocal`` keyword.

    In Python2 however, this attribute is read only, and there is no
    ``nonlocal`` keyword. For this reason, we need to come up with more
    complicated hacks to set this attribute.

    The chosen approach is to create a function with a STORE_DEREF opcode,
    which sets the content of a closure variable. Typically:

    >>> def inner(value):
    ...     lambda: cell  # the lambda makes cell a closure
    ...     cell = value  # cell is a closure, so this triggers a STORE_DEREF

    (Note that in Python2, A STORE_DEREF can never be triggered from an inner
    function. The function g for example here
    >>> def f(var):
    ...     def g():
    ...         var += 1
    ...     return g

    will not modify the closure variable ``var```inplace, but instead try to
    load a local variable var and increment it. As g does not assign the local
    variable ``var`` any initial value, calling f(1)() will fail at runtime.)

    Our objective is to set the value of a given cell ``cell``. So we need to
    somewhat reference our ``cell`` object into the ``inner`` function so that
    this object (and not the smoke cell of the lambda function) gets affected
    by the STORE_DEREF operation.

    In inner, ``cell`` is referenced as a cell variable (an enclosing variable
    that is referenced by the inner function). If we create a new function
    cell_set with the exact same code as ``inner``, but with ``cell`` marked as
    a free variable instead, the STORE_DEREF will be applied on its closure -
    ``cell``, which we can specify explicitly during construction! The new
    cell_set variable thus actually sets the contents of a specified cell!

    Note: we do not make use of the ``nonlocal`` keyword to set the contents of
    a cell in early python3 versions to limit possible syntax errors in case
    test and checker libraries decide to parse the whole file.
    """
    if sys.version_info[:2] >= (3, 7):
        cell.cell_contents = value
    else:
        _cell_set = types.FunctionType(_cell_set_template_code, {}, '_cell_set', (), (cell,))
        _cell_set(value)