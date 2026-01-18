from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def boolean_arguments(*fields):

    def decorator(func):

        def wrapper(*args, **kw):
            for field in [f for f in fields if isinstance(kw.get(f), bool)]:
                kw[field] = str(kw[field]).lower()
            return func(*args, **kw)
        wrapper.__doc__ = '{0}\nBooleans: {1}'.format(func.__doc__, ', '.join(fields))
        return add_attrs_from(func, to=wrapper)
    return decorator