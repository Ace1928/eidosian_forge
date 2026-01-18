import codecs
import re
from typing import (IO, Iterator, Match, NamedTuple, Optional,  # noqa:F401
def make_regex(string: str, extra_flags: int=0) -> Pattern[str]:
    return re.compile(string, re.UNICODE | extra_flags)