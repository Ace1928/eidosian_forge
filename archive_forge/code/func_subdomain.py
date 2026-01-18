from pprint import pformat
from six import iteritems
import re
@subdomain.setter
def subdomain(self, subdomain):
    """
        Sets the subdomain of this V1PodSpec.
        If specified, the fully qualified Pod hostname will be
        "<hostname>.<subdomain>.<pod namespace>.svc.<cluster domain>". If not
        specified, the pod will not have a domainname at all.

        :param subdomain: The subdomain of this V1PodSpec.
        :type: str
        """
    self._subdomain = subdomain