import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def collapse_white_spaces(s: str) -> str:
    return WHITESPACES_PATTERN.sub(' ', s).strip(' ')