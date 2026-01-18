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
def structured_objects(*fields, **kwargs):

    def decorator(func):

        def wrapper(*args, **kw):
            members = kwargs.get('members', False)
            for field in filter(lambda i: i in kw, fields):
                destructure_object(kw.pop(field), kw, field, members=members)
            return func(*args, **kw)
        wrapper.__doc__ = '{0}\nElement|Iter|Map: {1}\n(ResponseElement or anything iterable/dict-like)'.format(func.__doc__, ', '.join(fields))
        return add_attrs_from(func, to=wrapper)
    return decorator