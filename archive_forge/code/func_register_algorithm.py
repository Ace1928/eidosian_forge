from __future__ import annotations
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any
from .algorithms import (
from .exceptions import (
from .utils import base64url_decode, base64url_encode
from .warnings import RemovedInPyjwt3Warning
def register_algorithm(self, alg_id: str, alg_obj: Algorithm) -> None:
    """
        Registers a new Algorithm for use when creating and verifying tokens.
        """
    if alg_id in self._algorithms:
        raise ValueError('Algorithm already has a handler.')
    if not isinstance(alg_obj, Algorithm):
        raise TypeError('Object is not of type `Algorithm`')
    self._algorithms[alg_id] = alg_obj
    self._valid_algs.add(alg_id)