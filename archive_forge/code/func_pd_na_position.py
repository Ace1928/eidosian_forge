from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
@property
def pd_na_position(self) -> str:
    return self._metadata['pd_na_position']