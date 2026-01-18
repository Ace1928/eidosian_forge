from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
@classmethod
def fromRRHeader(cls, rrHeader):
    """
        A classmethod for constructing a new L{_OPTHeader} from the
        attributes and payload of an existing L{RRHeader} instance.

        @type rrHeader: L{RRHeader}
        @param rrHeader: An L{RRHeader} instance containing an
            L{UnknownRecord} payload.

        @return: An instance of L{_OPTHeader}.
        @rtype: L{_OPTHeader}
        """
    options = None
    if rrHeader.payload is not None:
        options = []
        optionsBytes = BytesIO(rrHeader.payload.data)
        optionsBytesLength = len(rrHeader.payload.data)
        while optionsBytes.tell() < optionsBytesLength:
            o = _OPTVariableOption()
            o.decode(optionsBytes)
            options.append(o)
    return cls(udpPayloadSize=rrHeader.cls, extendedRCODE=rrHeader.ttl >> 24, version=rrHeader.ttl >> 16 & 255, dnssecOK=(rrHeader.ttl & 65535) >> 15, options=options)