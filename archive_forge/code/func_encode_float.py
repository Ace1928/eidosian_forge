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
def encode_float(self, value):
    """
        See http://tools.ietf.org/html/draft-wood-ldapext-float-00.
        """
    s = '%e' % value
    l = s.split('e')
    mantissa = l[0].ljust(18, '0')
    exponent = l[1]
    if value == 0.0:
        case = '3'
        exponent = '000'
    elif mantissa[0] != '-' and exponent[0] == '+':
        case = '5'
        exponent = exponent[1:].rjust(3, '0')
    elif mantissa[0] != '-' and exponent[0] == '-':
        case = '4'
        exponent = 999 + int(exponent)
        exponent = '%03d' % exponent
    elif mantissa[0] == '-' and exponent[0] == '-':
        case = '2'
        mantissa = '%f' % (10 + float(mantissa))
        mantissa = mantissa.ljust(18, '0')
        exponent = exponent[1:].rjust(3, '0')
    else:
        case = '1'
        mantissa = '%f' % (10 + float(mantissa))
        mantissa = mantissa.ljust(18, '0')
        exponent = 999 - int(exponent)
        exponent = '%03d' % exponent
    return '%s %s %s' % (case, exponent, mantissa)