from __future__ import annotations
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any
from .algorithms import (
from .exceptions import (
from .utils import base64url_decode, base64url_encode
from .warnings import RemovedInPyjwt3Warning
def get_algorithms(self) -> list[str]:
    """
        Returns a list of supported values for the 'alg' parameter.
        """
    return list(self._valid_algs)