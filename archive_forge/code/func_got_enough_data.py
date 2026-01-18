from typing import Tuple, Union
from .big5freq import (
from .euckrfreq import (
from .euctwfreq import (
from .gb2312freq import (
from .jisfreq import (
from .johabfreq import JOHAB_TO_EUCKR_ORDER_TABLE
def got_enough_data(self) -> bool:
    return self._total_chars > self.ENOUGH_DATA_THRESHOLD