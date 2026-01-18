import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_add_vm_disk(self, vapp_or_vm_id, vm_disk_size):
    """
        Adds a virtual disk to the specified VM or VMs under the vApp. If the
        vapp_or_vm_id param represents a link to an vApp all VMs that are
        attached to this vApp will be modified.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If a
                                   vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    vm_disk_size: the disk capacity in GB that will be added
                                  to the specified VM or VMs
        :type       vm_disk_size: ``int``

        :rtype: ``None``
        """
    self._validate_vm_disk_size(vm_disk_size)
    self._add_vm_disk(vapp_or_vm_id, vm_disk_size)