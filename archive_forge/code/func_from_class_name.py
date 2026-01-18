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
@staticmethod
def from_class_name(name: str, theme_element: Any) -> themeable:
    """
        Create a themeable by name

        Parameters
        ----------
        name : str
            Class name
        theme_element : element object
            An element of the type required by the theme.
            For lines, text and rects it should be one of:
            [](`~plotnine.themes.element_line`),
            [](`~plotnine.themes.element_rect`),
            [](`~plotnine.themes.element_text`) or
            [](`~plotnine.themes.element_blank`)

        Returns
        -------
        out : plotnine.themes.themeable.themeable
        """
    msg = f'There no themeable element called: {name}'
    try:
        klass: Type[themeable] = themeable._registry[name]
    except KeyError as e:
        raise PlotnineError(msg) from e
    if not issubclass(klass, themeable):
        raise PlotnineError(msg)
    return klass(theme_element)