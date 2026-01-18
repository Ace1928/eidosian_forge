from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def should_fail_under(total: float, fail_under: float, precision: int) -> bool:
    """Determine if a total should fail due to fail-under.

    `total` is a float, the coverage measurement total. `fail_under` is the
    fail_under setting to compare with. `precision` is the number of digits
    to consider after the decimal point.

    Returns True if the total should fail.

    """
    if not 0 <= fail_under <= 100.0:
        msg = f'fail_under={fail_under} is invalid. Must be between 0 and 100.'
        raise ConfigError(msg)
    if fail_under == 100.0 and total != 100.0:
        return True
    return round(total, precision) < fail_under