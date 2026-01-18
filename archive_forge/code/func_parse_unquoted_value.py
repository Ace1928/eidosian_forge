import codecs
import re
from typing import (IO, Iterator, Match, NamedTuple, Optional,  # noqa:F401
def parse_unquoted_value(reader: Reader) -> str:
    part, = reader.read_regex(_unquoted_value)
    return re.sub('\\s+#.*', '', part).rstrip()