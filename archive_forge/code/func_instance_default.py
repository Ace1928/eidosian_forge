from __future__ import annotations
import logging # isort:skip
from copy import copy
from types import FunctionType
from typing import (
from ...util.deprecation import deprecated
from .singletons import Undefined
from .wrappers import PropertyValueColumnData, PropertyValueContainer
def instance_default(self, obj: HasProps) -> T:
    """ Get the default value that will be used for a specific instance.

        Args:
            obj (HasProps) : The instance to get the default value for.

        Returns:
            object

        """
    return self.property.themed_default(obj.__class__, self.name, obj.themed_values())