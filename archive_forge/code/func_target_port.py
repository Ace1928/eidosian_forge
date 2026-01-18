from pprint import pformat
from six import iteritems
import re
@target_port.setter
def target_port(self, target_port):
    """
        Sets the target_port of this V1ServicePort.
        Number or name of the port to access on the pods targeted by the
        service. Number must be in the range 1 to 65535. Name must be an
        IANA_SVC_NAME. If this is a string, it will be looked up as a named port
        in the target Pod's container ports. If this is not specified, the value
        of the 'port' field is used (an identity map). This field is ignored for
        services with clusterIP=None, and should be omitted or set equal to the
        'port' field. More info:
        https://kubernetes.io/docs/concepts/services-networking/service/#defining-a-service

        :param target_port: The target_port of this V1ServicePort.
        :type: object
        """
    self._target_port = target_port