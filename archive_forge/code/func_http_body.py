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
def http_body(field):

    def decorator(func):

        def wrapper(*args, **kw):
            if any([f not in kw for f in (field, 'content_type')]):
                message = '{0} requires {1} and content_type arguments for building HTTP body'.format(func.action, field)
                raise KeyError(message)
            kw['body'] = kw.pop(field)
            kw['headers'] = {'Content-Type': kw.pop('content_type'), 'Content-MD5': content_md5(kw['body'])}
            return func(*args, **kw)
        wrapper.__doc__ = '{0}\nRequired HTTP Body: {1}'.format(func.__doc__, field)
        return add_attrs_from(func, to=wrapper)
    return decorator