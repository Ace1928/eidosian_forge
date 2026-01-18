import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_rebuild(self, node, image, **kwargs):
    """
        Rebuild a Node.

        :param      node: Node to rebuild.
        :type       node: :class:`Node`

        :param      image: New image to use.
        :type       image: :class:`NodeImage`

        :keyword    ex_metadata: Key/Value metadata to associate with a node
        :type       ex_metadata: ``dict``

        :keyword    ex_files:   File Path => File contents to create on
                                the node
        :type       ex_files:   ``dict``

        :keyword    ex_keyname:  Name of existing public key to inject into
                                 instance
        :type       ex_keyname:  ``str``

        :keyword    ex_userdata: String containing user data
                                 see
                                 https://help.ubuntu.com/community/CloudInit
        :type       ex_userdata: ``str``

        :keyword    ex_security_groups: List of security groups to assign to
                                        the node
        :type       ex_security_groups: ``list`` of
                                       :class:`OpenStackSecurityGroup`

        :keyword    ex_disk_config: Name of the disk configuration.
                                    Can be either ``AUTO`` or ``MANUAL``.
        :type       ex_disk_config: ``str``

        :keyword    ex_config_drive: If True enables metadata injection in a
                                     server through a configuration drive.
        :type       ex_config_drive: ``bool``

        :rtype: ``bool``
        """
    server_params = self._create_args_to_params(node, image=image, **kwargs)
    resp = self._node_action(node, 'rebuild', **server_params)
    return resp.status == httplib.ACCEPTED