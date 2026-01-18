from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def class_default(self, cls: type[HasProps], *, no_eval: bool=False):
    """ Get the default value for a specific subtype of ``HasProps``,
        which may not be used for an individual instance.

        Args:
            cls (class) : The class to get the default value for.

            no_eval (bool, optional) :
                Whether to evaluate callables for defaults (default: False)

        Returns:
            object


        """
    return self.property.themed_default(cls, self.name, None, no_eval=no_eval)