from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
@property
def has_partition_by(self) -> bool:
    return len(self.window.partition_keys) > 0