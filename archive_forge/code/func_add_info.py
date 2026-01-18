import abc
import base64
import json
import logging
import os
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery._utils import b64decode
from pymacaroons.serializers import json_serializer
from ._versions import (
from ._error import (
from ._codec import (
from ._keys import PublicKey
from ._third_party import (
def add_info(self, loc, info):
    """Associates the given information with the given location.
        It will ignore any trailing slash.
        @param loc the location as string
        @param info (ThirdPartyInfo) to store for this location.
        """
    self._store[loc.rstrip('/')] = info