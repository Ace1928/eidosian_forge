import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def split_function_test(function_test: str) -> List[str]:
    if not function_test.startswith('function('):
        return []
    elif function_test == 'function(*)':
        return ['*']
    parts = function_test[9:].partition(') as ')
    if parts[0]:
        sequence_types = parts[0].split(', ')
        sequence_types.append(parts[2])
    else:
        sequence_types = [parts[2]]
    return sequence_types