from __future__ import unicode_literals
import copy
from base64 import standard_b64encode
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.serializers.binary_serializer import BinarySerializer
from pymacaroons.exceptions import MacaroonInitException
from pymacaroons.utils import (
from pymacaroons.caveat_delegates import (
@property
def identifier_bytes(self):
    return self._identifier