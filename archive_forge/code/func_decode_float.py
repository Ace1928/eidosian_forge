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
def decode_float(self, value):
    case = value[0]
    exponent = value[2:5]
    mantissa = value[6:]
    if case == '3':
        return 0.0
    elif case == '5':
        pass
    elif case == '4':
        exponent = '%03d' % (int(exponent) - 999)
    elif case == '2':
        mantissa = '%f' % (float(mantissa) - 10)
        exponent = '-' + exponent
    else:
        mantissa = '%f' % (float(mantissa) - 10)
        exponent = '%03d' % abs(int(exponent) - 999)
    return float(mantissa + 'e' + exponent)