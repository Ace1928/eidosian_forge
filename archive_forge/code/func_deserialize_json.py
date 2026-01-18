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
@classmethod
def deserialize_json(cls, serialized_json):
    """Return a macaroon deserialized from a string
        @param serialized_json The string to decode {str}
        @return {Macaroon}
        """
    serialized = json.loads(serialized_json)
    return Macaroon.from_dict(serialized)