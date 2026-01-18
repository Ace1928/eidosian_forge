import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
@property
def nanosecond(self):
    """Read-only: nanosecond precision."""
    return self._nanosecond