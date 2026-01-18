from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
def is_blank(self, name: str) -> bool:
    """
        Return True if the themeable *name* is blank

        If the *name* is not in the list of themeables then
        the lookup falls back to inheritance hierarchy.
        If none of the themeables are in the hierarchy are
        present, `False` is returned.

        Parameters
        ----------
        names : str
            Themeable, in order of most specific to most
            general.
        """
    for th in themeable._hierarchy[name]:
        if (element := self.get(th)):
            return element.is_blank()
    return False