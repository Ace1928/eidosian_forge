from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@classmethod
def fromduration(cls, duration: 'Duration') -> 'Timezone':
    if duration.seconds % 60 != 0:
        raise ValueError('{!r} has not an integral number of minutes'.format(duration))
    return cls(datetime.timedelta(seconds=int(duration.seconds)))