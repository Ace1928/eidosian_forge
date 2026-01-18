import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
@classmethod
def from_rfc3339(cls, stamp):
    """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (str): RFC3339 stamp, with up to nanosecond precision

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp string

        Raises:
            ValueError: if `stamp` does not match the expected format
        """
    with_nanos = _RFC3339_NANOS.match(stamp)
    if with_nanos is None:
        raise ValueError('Timestamp: {}, does not match pattern: {}'.format(stamp, _RFC3339_NANOS.pattern))
    bare = datetime.datetime.strptime(with_nanos.group('no_fraction'), _RFC3339_NO_FRACTION)
    fraction = with_nanos.group('nanos')
    if fraction is None:
        nanos = 0
    else:
        scale = 9 - len(fraction)
        nanos = int(fraction) * 10 ** scale
    return cls(bare.year, bare.month, bare.day, bare.hour, bare.minute, bare.second, nanosecond=nanos, tzinfo=datetime.timezone.utc)