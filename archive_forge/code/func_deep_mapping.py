import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
def deep_mapping(key_validator, value_validator, mapping_validator=None):
    """
    A validator that performs deep validation of a dictionary.

    :param key_validator: Validator to apply to dictionary keys
    :param value_validator: Validator to apply to dictionary values
    :param mapping_validator: Validator to apply to top-level mapping
        attribute (optional)

    .. versionadded:: 19.1.0

    :raises TypeError: if any sub-validators fail
    """
    return _DeepMapping(key_validator, value_validator, mapping_validator)