import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_restore(self, node, template=None):
    """
        Restore virtual machine

        :param node: Node to restore
        :type node: :class:`CloudStackNode`

        :param template: Optional new template
        :type  template: :class:`NodeImage`

        :rtype ``str``
        """
    params = {'virtualmachineid': node.id}
    if template:
        params['templateid'] = template.id
    res = self._async_request(command='restoreVirtualMachine', params=params, method='GET')
    return res['virtualmachine']['templateid']