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
def ex_change_vm_admin_password(self, vapp_or_vm_id, ex_admin_password):
    """
        Changes the admin (or root) password of VM or VMs under the vApp. If
        the vapp_or_vm_id param represents a link to an vApp all VMs that
        are attached to this vApp will be modified.

        :keyword    vapp_or_vm_id: vApp or VM ID that will be modified. If a
                                   vApp ID is used here all attached VMs
                                   will be modified
        :type       vapp_or_vm_id: ``str``

        :keyword    ex_admin_password: admin password to be used.
        :type       ex_admin_password: ``str``

        :rtype: ``None``
        """
    if ex_admin_password is None:
        return
    vms = self._get_vm_elements(vapp_or_vm_id)
    for vm in vms:
        res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')))
        headers = {'Content-Type': 'application/vnd.vmware.vcloud.guestCustomizationSection+xml'}
        auto_logon = res.object.find(fixxpath(res.object, 'AdminAutoLogonEnabled'))
        if auto_logon is not None and auto_logon.text == 'false':
            self._update_or_insert_section(res, 'AdminAutoLogonCount', 'ResetPasswordRequired', '0')
        self._update_or_insert_section(res, 'AdminPasswordAuto', 'AdminPassword', 'false')
        self._update_or_insert_section(res, 'AdminPasswordEnabled', 'AdminPasswordAuto', 'true')
        self._update_or_insert_section(res, 'AdminPassword', 'AdminAutoLogonEnabled', ex_admin_password)
        res = self.connection.request('%s/guestCustomizationSection' % get_url_path(vm.get('href')), data=ET.tostring(res.object), method='PUT', headers=headers)
        self._wait_for_task_completion(res.object.get('href'))