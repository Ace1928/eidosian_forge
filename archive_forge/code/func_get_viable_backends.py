from __future__ import annotations
import os
import abc
import logging
import operator
import copy
import typing
from .py312compat import metadata
from . import credentials, errors, util
from ._compat import properties
@classmethod
def get_viable_backends(cls: typing.Type[KeyringBackend]) -> filter[typing.Type[KeyringBackend]]:
    """
        Return all subclasses deemed viable.
        """
    return filter(operator.attrgetter('viable'), cls._classes)