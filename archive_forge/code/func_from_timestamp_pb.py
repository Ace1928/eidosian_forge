import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
@classmethod
def from_timestamp_pb(cls, stamp):
    """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (:class:`~google.protobuf.timestamp_pb2.Timestamp`): timestamp message

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp message
        """
    microseconds = int(stamp.seconds * 1000000.0)
    bare = from_microseconds(microseconds)
    return cls(bare.year, bare.month, bare.day, bare.hour, bare.minute, bare.second, nanosecond=stamp.nanos, tzinfo=datetime.timezone.utc)