from pprint import pformat
from six import iteritems
import re
@client_ip.setter
def client_ip(self, client_ip):
    """
        Sets the client_ip of this V1SessionAffinityConfig.
        clientIP contains the configurations of Client IP based session
        affinity.

        :param client_ip: The client_ip of this V1SessionAffinityConfig.
        :type: V1ClientIPConfig
        """
    self._client_ip = client_ip