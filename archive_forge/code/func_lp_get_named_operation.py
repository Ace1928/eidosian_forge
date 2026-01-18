from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
def lp_get_named_operation(self, operation_name):
    """Get a custom operation with the given name.

        :return: A NamedOperation instance that can be called with
                 appropriate arguments to invoke the operation.
        """
    params = {'ws.op': operation_name}
    method = self._wadl_resource.get_method('get', query_params=params)
    if method is None:
        method = self._wadl_resource.get_method('post', representation_params=params)
    if method is None:
        raise KeyError('No operation with name: %s' % operation_name)
    return NamedOperation(self._root, self, method)