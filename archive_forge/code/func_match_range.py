from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_range(self, el: bs4.Tag, condition: int) -> bool:
    """
        Match range.

        Behavior is modeled after what we see in browsers. Browsers seem to evaluate
        if the value is out of range, and if not, it is in range. So a missing value
        will not evaluate out of range; therefore, value is in range. Personally, I
        feel like this should evaluate as neither in or out of range.
        """
    out_of_range = False
    itype = util.lower(self.get_attribute_by_name(el, 'type'))
    mn = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'min', None)))
    mx = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'max', None)))
    if mn is None and mx is None:
        return False
    value = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'value', None)))
    if value is not None:
        if itype in ('date', 'datetime-local', 'month', 'week', 'number', 'range'):
            if mn is not None and value < mn:
                out_of_range = True
            if not out_of_range and mx is not None and (value > mx):
                out_of_range = True
        elif itype == 'time':
            if mn is not None and mx is not None and (mn > mx):
                if value < mn and value > mx:
                    out_of_range = True
            else:
                if mn is not None and value < mn:
                    out_of_range = True
                if not out_of_range and mx is not None and (value > mx):
                    out_of_range = True
    return not out_of_range if condition & ct.SEL_IN_RANGE else out_of_range