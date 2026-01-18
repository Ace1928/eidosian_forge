from typing import Dict, List, Optional, Type
from .. import config
from ..utils import logging
from .formatting import (
from .np_formatter import NumpyFormatter
def get_formatter(format_type: Optional[str], **format_kwargs) -> Formatter:
    """
    Factory function to get a Formatter given its type name and keyword arguments.
    A formatter is an object that extracts and formats data from pyarrow table.
    It defines the formatting for rows, colums and batches.
    If the formatter for a given type name doesn't exist or is not available, an error is raised.
    """
    format_type = get_format_type_from_alias(format_type)
    if format_type in _FORMAT_TYPES:
        return _FORMAT_TYPES[format_type](**format_kwargs)
    if format_type in _FORMAT_TYPES_ALIASES_UNAVAILABLE:
        raise _FORMAT_TYPES_ALIASES_UNAVAILABLE[format_type]
    else:
        raise ValueError(f"Return type should be None or selected in {list((type for type in _FORMAT_TYPES.keys() if type != None))}, but got '{format_type}'")