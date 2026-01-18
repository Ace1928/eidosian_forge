import inspect
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Type, TypeVar, Union, get_args
@classmethod
def parse_obj_as_instance(cls: Type[T], data: Union[bytes, str, List, Dict]) -> T:
    """Alias to parse server response and return a single instance.

        See `parse_obj` for more details.
        """
    output = cls.parse_obj(data)
    if isinstance(output, list):
        raise ValueError(f'Invalid input data for {cls}. Expected a single instance, but got a list.')
    return output