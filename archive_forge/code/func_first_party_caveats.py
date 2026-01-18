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
def first_party_caveats(self):
    """Return the first party caveats from this macaroon.

        @return the first party caveats from this macaroon as pymacaroons
        caveats.
        """
    return self._macaroon.first_party_caveats()