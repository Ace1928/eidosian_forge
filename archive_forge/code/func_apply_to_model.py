from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, overload
import yaml
from ..core.has_props import HasProps
from ..core.types import PathLike
from ..util.deprecation import deprecated
def apply_to_model(self, model: Model) -> None:
    """ Apply this theme to a model.

        .. warning::
            Typically, don't call this method directly. Instead, set the theme
            on the |Document| the model is a part of.

        """
    model.apply_theme(self._for_class(model.__class__))
    if len(_empty_dict) > 0:
        raise RuntimeError('Somebody put stuff in _empty_dict')