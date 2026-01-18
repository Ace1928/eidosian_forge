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
def api_action(section, quota, restore, *api):

    def decorator(func, quota=int(quota), restore=float(restore)):
        version, accesskey, path = api_version_path[section]
        action = ''.join(api or map(str.capitalize, func.__name__.split('_')))

        def wrapper(self, *args, **kw):
            kw.setdefault(accesskey, getattr(self, accesskey, None))
            if kw[accesskey] is None:
                message = '{0} requires {1} argument. Set the MWSConnection.{2} attribute?'.format(action, accesskey, accesskey)
                raise KeyError(message)
            kw['Action'] = action
            kw['Version'] = version
            response = self._response_factory(action, connection=self)
            request = dict(path=path, quota=quota, restore=restore)
            return func(self, request, response, *args, **kw)
        for attr in decorated_attrs:
            setattr(wrapper, attr, locals().get(attr))
        wrapper.__doc__ = 'MWS {0}/{1} API call; quota={2} restore={3:.2f}\n{4}'.format(action, version, quota, restore, func.__doc__)
        api_call_map[action] = func.__name__
        return wrapper
    return decorator