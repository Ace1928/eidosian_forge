import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def match_wildcard(name: str, wildcard: str) -> bool:
    if wildcard == '*' or wildcard == '*:*':
        return True
    elif wildcard.startswith('*:'):
        if name.startswith('{'):
            return name.endswith(f'}}{wildcard[2:]}')
        else:
            return name == wildcard[2:]
    elif wildcard.startswith('{') and wildcard.endswith('}*') or wildcard.endswith(':*'):
        return name.startswith(wildcard[:-1])
    else:
        return False