import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
def deep_iterable(member_validator, iterable_validator=None):
    """
    A validator that performs deep validation of an iterable.

    :param member_validator: Validator(s) to apply to iterable members
    :param iterable_validator: Validator to apply to iterable itself
        (optional)

    .. versionadded:: 19.1.0

    :raises TypeError: if any sub-validators fail
    """
    if isinstance(member_validator, (list, tuple)):
        member_validator = and_(*member_validator)
    return _DeepIterable(member_validator, iterable_validator)