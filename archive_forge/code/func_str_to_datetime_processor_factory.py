from __future__ import annotations
import datetime
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from datetime import time as time_cls
from decimal import Decimal
import typing
from typing import Any
from typing import Callable
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
def str_to_datetime_processor_factory(regexp: typing.Pattern[str], type_: Callable[..., _DT]) -> Callable[[Optional[str]], Optional[_DT]]:
    rmatch = regexp.match
    has_named_groups = bool(regexp.groupindex)

    def process(value: Optional[str]) -> Optional[_DT]:
        if value is None:
            return None
        else:
            try:
                m = rmatch(value)
            except TypeError as err:
                raise ValueError("Couldn't parse %s string '%r' - value is not a string." % (type_.__name__, value)) from err
            if m is None:
                raise ValueError("Couldn't parse %s string: '%s'" % (type_.__name__, value))
            if has_named_groups:
                groups = m.groupdict(0)
                return type_(**dict(list(zip(iter(groups.keys()), list(map(int, iter(groups.values())))))))
            else:
                return type_(*list(map(int, m.groups(0))))
    return process