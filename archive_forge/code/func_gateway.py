from pprint import pformat
from six import iteritems
import re
@gateway.setter
def gateway(self, gateway):
    """
        Sets the gateway of this V1ScaleIOVolumeSource.
        The host address of the ScaleIO API Gateway.

        :param gateway: The gateway of this V1ScaleIOVolumeSource.
        :type: str
        """
    if gateway is None:
        raise ValueError('Invalid value for `gateway`, must not be `None`')
    self._gateway = gateway