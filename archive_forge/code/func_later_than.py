import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def later_than(after, before):
    """True if then is later or equal to that"""
    if isinstance(after, str):
        after = str_to_time(after)
    elif isinstance(after, int):
        after = time.gmtime(after)
    if isinstance(before, str):
        before = str_to_time(before)
    elif isinstance(before, int):
        before = time.gmtime(before)
    if before is None:
        return True
    if after is None:
        return False
    return after >= before