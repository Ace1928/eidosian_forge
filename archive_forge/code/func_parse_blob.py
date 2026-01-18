from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def parse_blob():
    return lambda stream: [read_blob(stream, struct_parse(structs.Dwarf_uleb128(''), stream))]