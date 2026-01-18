from __future__ import annotations
import inspect
import re
import types
from typing import Any
def class_of(value: Any) -> Any:
    """Returns a string of the value's type with an indefinite article.

    For example 'an Image' or 'a PlotValue'.
    """
    if inspect.isclass(value):
        return add_article(value.__name__)
    else:
        return class_of(type(value))