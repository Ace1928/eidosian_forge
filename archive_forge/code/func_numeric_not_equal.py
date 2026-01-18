import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def numeric_not_equal(op1: SupportsFloat, op2: SupportsFloat) -> bool:
    if op1 == op2:
        return False
    return not math.isclose(op1, op2, rel_tol=1e-07, abs_tol=0.0)