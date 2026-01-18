from __future__ import annotations
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any
from .algorithms import (
from .exceptions import (
from .utils import base64url_decode, base64url_encode
from .warnings import RemovedInPyjwt3Warning
def get_algorithm_by_name(self, alg_name: str) -> Algorithm:
    """
        For a given string name, return the matching Algorithm object.

        Example usage:

        >>> jws_obj.get_algorithm_by_name("RS256")
        """
    try:
        return self._algorithms[alg_name]
    except KeyError as e:
        if not has_crypto and alg_name in requires_cryptography:
            raise NotImplementedError(f"Algorithm '{alg_name}' could not be found. Do you have cryptography installed?") from e
        raise NotImplementedError('Algorithm not supported') from e