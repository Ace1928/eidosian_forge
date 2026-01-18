import inspect
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Type, TypeVar, Union, get_args
@classmethod
def parse_obj_as_list(cls: Type[T], data: Union[bytes, str, List, Dict]) -> List[T]:
    """Alias to parse server response and return a single instance.

        See `parse_obj` for more details.
        """
    output = cls.parse_obj(data)
    if not isinstance(output, list):
        raise ValueError(f'Invalid input data for {cls}. Expected a list, but got {type(output)}.')
    return output