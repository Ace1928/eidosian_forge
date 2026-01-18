import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def ex_node_set_save_name(self, node, name):
    """
        Build action representation and instruct node to commit action.

        Build action representation from the compute node ID, the disk image
        which will be saved, and the name under which the image will be saved
        upon shutting down the compute node.

        :param node: Compute node instance.
        :type  node: :class:`Node`

        :param name: Name under which the image should be saved after shutting
                     down the compute node.
        :type  name: ``str``

        :return: False if an HTTP Bad Request is received, else, True is
                 returned.
        :rtype:  ``bool``
        """
    compute_node_id = str(node.id)
    compute = ET.Element('COMPUTE')
    compute_id = ET.SubElement(compute, 'ID')
    compute_id.text = compute_node_id
    disk = ET.SubElement(compute, 'DISK', {'id': str(node.image.id)})
    ET.SubElement(disk, 'STORAGE', {'href': '/storage/%s' % str(node.image.id), 'name': node.image.name})
    ET.SubElement(disk, 'SAVE_AS', {'name': str(name)})
    xml = ET.tostring(compute)
    url = '/compute/%s' % compute_node_id
    resp = self.connection.request(url, method='PUT', data=xml)
    if resp.status == httplib.BAD_REQUEST:
        return False
    else:
        return True