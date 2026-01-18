import inspect
from collections import namedtuple
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from warnings import warn
from .. import get_config
from ..exceptions import UnsetMetadataPassedError
from ._bunch import Bunch
def request_is_valid(item):
    """Check if an item is a valid request value (and not an alias).

    Parameters
    ----------
    item : object
        The given item to be checked.

    Returns
    -------
    result : bool
        Whether the given item is valid.
    """
    return item in VALID_REQUEST_VALUES