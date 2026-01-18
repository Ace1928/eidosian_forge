import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
def from_microseconds(value):
    """Convert timestamp in microseconds since the unix epoch to datetime.

    Args:
        value (float): The timestamp to convert, in microseconds.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp in
            UTC.
    """
    return _UTC_EPOCH + datetime.timedelta(microseconds=value)