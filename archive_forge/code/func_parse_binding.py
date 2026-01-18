import codecs
import re
from typing import (IO, Iterator, Match, NamedTuple, Optional,  # noqa:F401
def parse_binding(reader: Reader) -> Binding:
    reader.set_mark()
    try:
        reader.read_regex(_multiline_whitespace)
        if not reader.has_next():
            return Binding(key=None, value=None, original=reader.get_marked(), error=False)
        reader.read_regex(_export)
        key = parse_key(reader)
        reader.read_regex(_whitespace)
        if reader.peek(1) == '=':
            reader.read_regex(_equal_sign)
            value: Optional[str] = parse_value(reader)
        else:
            value = None
        reader.read_regex(_comment)
        reader.read_regex(_end_of_line)
        return Binding(key=key, value=value, original=reader.get_marked(), error=False)
    except Error:
        reader.read_regex(_rest_of_line)
        return Binding(key=None, value=None, original=reader.get_marked(), error=True)