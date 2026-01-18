from __future__ import annotations
import functools
import warnings
from textwrap import dedent
from typing import Optional, Type, Union
def deprecated_themeable_name(cls):
    """
    Decorator to deprecate the name of a themeable
    """
    old_init = cls.__init__

    @functools.wraps(cls.__init__)
    def new_init(self, *args, **kwargs):
        old_name = cls.__name__
        new_name = cls.mro()[1].__name__
        msg = f"\nThemeable '{old_name}' has been renamed to '{new_name}'.\n'{old_name}' is now deprecated and will be removed in a future release."
        warnings.warn(msg, category=FutureWarning, stacklevel=4)
        old_init(self, *args, **kwargs)
    cls.__init__ = new_init
    return cls