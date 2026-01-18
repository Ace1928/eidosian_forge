import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import to_bytes
from pyasn1.compat.octets import (int2oct, oct2int, ints2octs, null,
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
def encodeTag(self, singleTag, isConstructed):
    tagClass, tagFormat, tagId = singleTag
    encodedTag = tagClass | tagFormat
    if isConstructed:
        encodedTag |= tag.tagFormatConstructed
    if tagId < 31:
        return (encodedTag | tagId,)
    else:
        substrate = (tagId & 127,)
        tagId >>= 7
        while tagId:
            substrate = (128 | tagId & 127,) + substrate
            tagId >>= 7
        return (encodedTag | 31,) + substrate