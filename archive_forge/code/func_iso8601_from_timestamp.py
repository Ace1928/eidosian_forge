import datetime
import iso8601
from oslo_utils import encodeutils
def iso8601_from_timestamp(timestamp, microsecond=False):
    """Returns an iso8601 formatted date from timestamp."""
    return isotime(datetime.datetime.utcfromtimestamp(timestamp), microsecond)