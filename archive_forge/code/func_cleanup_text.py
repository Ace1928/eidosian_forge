import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def cleanup_text(raw_text: str) -> str:
    if '\\' in raw_text:
        raw_text = re.sub(_control_char_regexp, lambda s: _control_char_map[s[0]], raw_text)
    return raw_text.replace("''", "'")