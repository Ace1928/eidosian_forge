import datetime
import os
import re
from oslo_serialization import jsonutils as json
from blazarclient import exception
from blazarclient.i18n import _
def from_elapsed_time_to_delta(elapsed_time, pos_sign=True):
    """Return the positive or negative delta time based on the
    elapsed_time parameter.
    :param: elapsed_time: a string that matches ELAPSED_TIME_REGEX
    :param: sign: if sign is True, the returned value will be negative.
    Otherwise it will be positive.
    """
    seconds = from_elapsed_time_to_seconds(elapsed_time, pos_sign=pos_sign)
    return datetime.timedelta(seconds=seconds)