import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def request_is_alias(item):
    """Check if an item is a valid alias.

    Values in ``VALID_REQUEST_VALUES`` are not considered aliases in this
    context. Only a string which is a valid identifier is.

    Parameters
    ----------
    item : object
        The given item to be checked if it can be an alias.

    Returns
    -------
    result : bool
        Whether the given item is a valid alias.
    """
    if item in VALID_REQUEST_VALUES:
        return False
    return isinstance(item, str) and item.isidentifier()