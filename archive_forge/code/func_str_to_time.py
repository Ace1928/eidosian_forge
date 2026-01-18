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
def str_to_time(value: Optional[str]) -> Optional[datetime.time]:
    if value is not None:
        dt_value = time_cls.fromisoformat(value)
    else:
        dt_value = None
    return dt_value