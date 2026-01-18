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
def requires_some_of(*fields):

    def decorator(func):

        def requires(*args, **kw):
            if not any((i in kw for i in fields)):
                message = '{0} requires at least one of {1} argument(s)'.format(func.action, ', '.join(fields))
                raise KeyError(message)
            return func(*args, **kw)
        requires.__doc__ = '{0}\nSome Required: {1}'.format(func.__doc__, ', '.join(fields))
        return add_attrs_from(func, to=requires)
    return decorator