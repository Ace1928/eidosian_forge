from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_sample():
    import datetime
    type_map.update({'char': Char, 'Complex': complex, 'double': float, 'Foo.HANDLE': int, 'HANDLE': int, 'Null': None, 'nullptr': None, 'ObjectType.Identifier': Missing('sample.ObjectType.Identifier'), 'OddBool': bool, 'PStr': str, 'PyDate': datetime.date, 'sample.bool': bool, 'sample.char': Char, 'sample.double': float, 'sample.int': int, 'sample.ObjectType': object, 'sample.OddBool': bool, 'sample.Photon.TemplateBase': Missing('sample.Photon.TemplateBase'), 'sample.Point': Point, 'sample.PStr': str, 'sample.unsigned char': Char, 'std.size_t': int, 'std.string': str, 'ZeroIn': 0, 'Str("<unk")': '<unk', 'Str("<unknown>")': '<unknown>', 'Str("nown>")': 'nown>'})
    return locals()