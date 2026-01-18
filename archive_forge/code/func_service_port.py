from pprint import pformat
from six import iteritems
import re
@service_port.setter
def service_port(self, service_port):
    """
        Sets the service_port of this NetworkingV1beta1IngressBackend.
        Specifies the port of the referenced service.

        :param service_port: The service_port of this
        NetworkingV1beta1IngressBackend.
        :type: object
        """
    if service_port is None:
        raise ValueError('Invalid value for `service_port`, must not be `None`')
    self._service_port = service_port