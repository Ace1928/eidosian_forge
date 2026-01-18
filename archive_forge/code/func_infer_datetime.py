import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def infer_datetime(ob: Union[AnyDatetime, StructDatetime]) -> datetime.datetime:
    if isinstance(ob, (time.struct_time, tuple)):
        ob = datetime.datetime(*ob[:6])
    return ensure_datetime(ob)