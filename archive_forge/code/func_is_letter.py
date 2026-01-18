import re
from typing import Dict
from isoduration.constants import PERIOD_PREFIX, TIME_PREFIX, WEEK_PREFIX
from isoduration.parser.exceptions import OutOfDesignators
def is_letter(ch: str) -> bool:
    return ch.isalpha() and ch.lower() != 'e'