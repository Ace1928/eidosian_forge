from __future__ import unicode_literals
import binascii
from collections import namedtuple
import six
import struct
import sys
from base64 import urlsafe_b64encode
from pymacaroons.utils import (
from pymacaroons.serializers.base_serializer import BaseSerializer
from pymacaroons.exceptions import MacaroonSerializationException
def deserialize_raw(self, serialized):
    from pymacaroons.macaroon import MACAROON_V2
    from pymacaroons.exceptions import MacaroonDeserializationException
    first = six.byte2int(serialized[:1])
    if first == MACAROON_V2:
        return self._deserialize_v2(serialized)
    if _is_ascii_hex(first):
        return self._deserialize_v1(serialized)
    raise MacaroonDeserializationException('cannot determine data format of binary-encoded macaroon')