import calendar
import datetime
import re
from cloudsdk.google.protobuf import timestamp_pb2
from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.
def rfc3339(self):
    """Return an RFC3339-compliant timestamp.

        Returns:
            (str): Timestamp string according to RFC3339 spec.
        """
    if self._nanosecond == 0:
        return to_rfc3339(self)
    nanos = str(self._nanosecond).rjust(9, '0').rstrip('0')
    return '{}.{}Z'.format(self.strftime(_RFC3339_NO_FRACTION), nanos)