import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def decode_time(self, value):
    """ converts strings in the form of HH:MM:SS.mmmmmm
            (created by datetime.time.isoformat()) to
            datetime.time objects.

            Timzone-aware strings ("HH:MM:SS.mmmmmm+HH:MM") won't
            be handled right now and will raise TimeDecodeError.
        """
    if '-' in value or '+' in value:
        raise TimeDecodeError("Can't handle timezone aware objects: %r" % value)
    tmp = value.split('.')
    arg = map(int, tmp[0].split(':'))
    if len(tmp) == 2:
        arg.append(int(tmp[1]))
    return time(*arg)