from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def parse_nestedexpr():

    def parse(stream):
        size = struct_parse(structs.Dwarf_uleb128(''), stream)
        nested_expr_blob = read_blob(stream, size)
        return [DWARFExprParser(structs).parse_expr(nested_expr_blob)]
    return parse