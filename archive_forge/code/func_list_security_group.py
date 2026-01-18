import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def list_security_group(self, server):
    """
        List Security Group(s) of an instance

        :param server: ID of the instance.

        """
    return self._list('/servers/%s/os-security-groups' % base.getid(server), 'security_groups', SecurityGroup)