import logging
import time
from monotonic import monotonic as now  # noqa
def pick_first_not_none(*values):
    """Returns first of values that is *not* None (or None if all are/were)."""
    for val in values:
        if val is not None:
            return val
    return None