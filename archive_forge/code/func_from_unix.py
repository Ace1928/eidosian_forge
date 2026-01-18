from collections import namedtuple
import datetime
import sys
import struct
@staticmethod
def from_unix(unix_sec):
    """Create a Timestamp from posix timestamp in seconds.

        :param unix_float: Posix timestamp in seconds.
        :type unix_float: int or float.
        """
    seconds = int(unix_sec // 1)
    nanoseconds = int(unix_sec % 1 * 10 ** 9)
    return Timestamp(seconds, nanoseconds)