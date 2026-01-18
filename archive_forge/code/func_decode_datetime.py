from __future__ import with_statement
import datetime
import functools
import inspect
import logging
import os
import re
import sys
import six
def decode_datetime(encoded_datetime, truncate_time=False):
    """Decode a DateTimeField parameter from a string to a python datetime.

    Args:
      encoded_datetime: A string in RFC 3339 format.
      truncate_time: If true, truncate time string with precision higher than
          microsecs.

    Returns:
      A datetime object with the date and time specified in encoded_datetime.

    Raises:
      ValueError: If the string is not in a recognized format.
    """
    time_zone_match = _TIME_ZONE_RE.search(encoded_datetime)
    if time_zone_match:
        time_string = encoded_datetime[:time_zone_match.start(1)].upper()
    else:
        time_string = encoded_datetime.upper()
    if '.' in time_string:
        format_string = '%Y-%m-%dT%H:%M:%S.%f'
    else:
        format_string = '%Y-%m-%dT%H:%M:%S'
    try:
        decoded_datetime = datetime.datetime.strptime(time_string, format_string)
    except ValueError:
        if truncate_time and '.' in time_string:
            datetime_string, decimal_secs = time_string.split('.')
            if len(decimal_secs) > 6:
                truncated_time_string = '{}.{}'.format(datetime_string, decimal_secs[:6])
                decoded_datetime = datetime.datetime.strptime(truncated_time_string, format_string)
                logging.warning('Truncating the datetime string from %s to %s', time_string, truncated_time_string)
            else:
                raise
        else:
            raise
    if not time_zone_match:
        return decoded_datetime
    if time_zone_match.group('z'):
        offset_minutes = 0
    else:
        sign = time_zone_match.group('sign')
        hours, minutes = [int(value) for value in time_zone_match.group('hours', 'minutes')]
        offset_minutes = hours * 60 + minutes
        if sign == '-':
            offset_minutes *= -1
    return datetime.datetime(decoded_datetime.year, decoded_datetime.month, decoded_datetime.day, decoded_datetime.hour, decoded_datetime.minute, decoded_datetime.second, decoded_datetime.microsecond, TimeZoneOffset(offset_minutes))