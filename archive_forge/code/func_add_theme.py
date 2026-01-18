from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def add_theme(self, other: theme) -> theme:
    """
        Add themes together

        Subclasses should not override this method.

        This will be called when adding two instances of class 'theme'
        together.
        A complete theme will annihilate any previous themes. Partial themes
        can be added together and can be added to a complete theme.
        """
    if other.complete:
        return other
    self.themeables.update(deepcopy(other.themeables))
    return self