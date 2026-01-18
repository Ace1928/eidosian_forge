import datetime
import time
from pytz import reference
def get_current_time_millis():
    """
    Returns the time in milliseconds since the epoch as an integer number.
    """
    return int(time.time() * 1000)