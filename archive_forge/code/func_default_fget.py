import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def default_fget(self, obj: Any) -> Union[Any, List[Any]]:
    if isinstance(self.path_or_name, str):
        return nested_get(obj, self.path_or_name)
    elif isinstance(self.path_or_name, list):
        return [nested_get(obj, p) for p in self.path_or_name]
    else:
        raise TypeError(f'Unexpected type for path {type(self.path_or_name)!r}')