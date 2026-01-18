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
@property
def lp_operations(self):
    """Name all of this resource's custom operations."""
    names = []
    for method in self._wadl_resource.method_iter:
        name = method.name.lower()
        if name == 'get':
            params = method.request.params(['query', 'plain'])
        elif name == 'post':
            for media_type in ['application/x-www-form-urlencoded', 'multipart/form-data']:
                definition = method.request.get_representation_definition(media_type)
                if definition is not None:
                    definition = definition.resolve_definition()
                    break
            params = definition.params(self._wadl_resource)
        for param in params:
            if param.name == 'ws.op':
                names.append(param.fixed_value)
                break
    return names