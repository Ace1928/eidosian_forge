import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def nested_set(json: dict, keys: str, value: Any) -> None:
    """Set the element at the terminal node of a nested JSON dict based on `path`.

    The first item of the path can be an object.

    If nodes do not exist, they are created along the way.
    """
    keys = keys.split('.')
    if len(keys) == 1:
        vars(json)[keys[0]] = value
    else:
        for key in keys[:-1]:
            if isinstance(json, Base):
                if not hasattr(json, key):
                    setattr(json, key, {})
                json = getattr(json, key)
            else:
                json = json.setdefault(key, {})
        json[keys[-1]] = value