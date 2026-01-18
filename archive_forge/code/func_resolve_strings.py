import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def resolve_strings(self, lineprog_header, format_field, data_field):
    if lineprog_header.get(format_field, False):
        data = lineprog_header[data_field]
        for field in lineprog_header[format_field]:

            def replace_value(data, content_type, replacer):
                for entry in data:
                    entry[content_type] = replacer(entry[content_type])
            if field.form == 'DW_FORM_line_strp':
                replace_value(data, field.content_type, self.get_string_from_linetable)
            elif field.form == 'DW_FORM_strp':
                replace_value(data, field.content_type, self.get_string_from_table)
            elif field.form in ('DW_FORM_strp_sup', 'DW_FORM_GNU_strp_alt'):
                if self.supplementary_dwarfinfo:
                    replace_value(data, field.content_type, self.supplementary_dwarfinfo.get_string_fromtable)
                else:
                    replace_value(data, field.content_type, lambda x: str(x))
            elif field.form in ('DW_FORM_strp_sup', 'DW_FORM_strx', 'DW_FORM_strx1', 'DW_FORM_strx2', 'DW_FORM_strx3', 'DW_FORM_strx4'):
                raise NotImplementedError()