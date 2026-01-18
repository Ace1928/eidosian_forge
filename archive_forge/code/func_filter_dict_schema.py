from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def filter_dict_schema(*, include: IncExDict | None=None, exclude: IncExDict | None=None) -> IncExDictSerSchema:
    return _dict_not_none(type='include-exclude-dict', include=include, exclude=exclude)