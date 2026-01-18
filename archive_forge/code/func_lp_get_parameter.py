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
def lp_get_parameter(self, param_name):
    """Get the value of one of the resource's parameters.

        :return: A scalar value if the parameter is not a link. A new
                 Resource object, whose resource is bound to a
                 representation, if the parameter is a link.
        """
    self._ensure_representation()
    for suffix in ['_link', '_collection_link']:
        param = self._wadl_resource.get_parameter(param_name + suffix)
        if param is not None:
            try:
                param.get_value()
            except KeyError:
                continue
            if param.get_value() is None:
                return None
            linked_resource = param.linked_resource
            return self._create_bound_resource(self._root, linked_resource, param_name=param.name)
    param = self._wadl_resource.get_parameter(param_name)
    if param is None:
        raise KeyError('No such parameter: %s' % param_name)
    return param.get_value()