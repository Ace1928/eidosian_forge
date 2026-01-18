from __future__ import annotations
import logging # isort:skip
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from ..util.serialization import make_id
from ..util.strings import append_docstring
def process_example(cls: type[Any]) -> None:
    """ A decorator to mark abstract base classes derived from |HasProps|.

    """
    if '__example__' in cls.__dict__:
        cls.__doc__ = append_docstring(cls.__doc__, _EXAMPLE_TEMPLATE.format(path=cls.__dict__['__example__']))