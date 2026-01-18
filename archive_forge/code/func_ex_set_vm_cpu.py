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
def ex_set_vm_cpu(self, vapp_or_vm_id, vm_cpu):
    """
        Sets the number of virtual CPUs for the specified VM or VMs under
        the vApp. If the vapp_or_vm_id param represents a link to an vApp
        all VMs that are attached to this vApp will be modified.

        Please ensure that hot-adding a virtual CPU is enabled for the
        powered on virtual machines. Otherwise use this method on undeployed
        vApp.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If
                                   a vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    vm_cpu: number of virtual CPUs/cores to allocate for
                            specified VMs
        :type       vm_cpu: ``int``

        :rtype: ``None``
        """
    self._validate_vm_cpu(vm_cpu)
    self._change_vm_cpu(vapp_or_vm_id, vm_cpu)