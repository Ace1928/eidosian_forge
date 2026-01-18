import datetime
import os
import re
from oslo_serialization import jsonutils as json
from blazarclient import exception
from blazarclient.i18n import _
def from_elapsed_time_to_seconds(elapsed_time, pos_sign=True):
    """Return the positive or negative amount of seconds based on the
    elapsed_time parameter with a sign depending on the sign parameter.
    :param: elapsed_time: a string that matches ELAPSED_TIME_REGEX
    :param: sign: if pos_sign is True, the returned value will be positive.
    Otherwise it will be positive.
    """
    is_elapsed_time = re.match(ELAPSED_TIME_REGEX, elapsed_time)
    if is_elapsed_time is None:
        raise exception.BlazarClientException(_('Invalid time format for option.'))
    elapsed_time_value = int(is_elapsed_time.group(1))
    elapsed_time_option = is_elapsed_time.group(2)
    seconds = {'s': lambda x: datetime.timedelta(seconds=x).total_seconds(), 'm': lambda x: datetime.timedelta(minutes=x).total_seconds(), 'h': lambda x: datetime.timedelta(hours=x).total_seconds(), 'd': lambda x: datetime.timedelta(days=x).total_seconds()}[elapsed_time_option](elapsed_time_value)
    if pos_sign:
        return int(seconds)
    return int(seconds) * -1