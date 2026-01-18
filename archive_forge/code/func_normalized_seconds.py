import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def normalized_seconds(seconds: Decimal) -> str:
    return '{:.6f}'.format(seconds).rstrip('0').rstrip('.')