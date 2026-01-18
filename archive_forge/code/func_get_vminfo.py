from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
def get_vminfo(self, node, vmid, **kwargs):
    global results
    results = {}
    mac = {}
    devices = {}
    try:
        vm = self.proxmox_api.nodes(node).qemu(vmid).config.get()
    except Exception as e:
        self.module.fail_json(msg='Getting information for VM with vmid = %s failed with exception: %s' % (vmid, e))
    kwargs = dict(((k, v) for k, v in kwargs.items() if v is not None))
    for k in list(kwargs.keys()):
        if isinstance(kwargs[k], dict):
            kwargs.update(kwargs[k])
            del kwargs[k]
    re_net = re.compile('net[0-9]')
    re_dev = re.compile('(virtio|ide|scsi|sata|efidisk)[0-9]')
    for k in kwargs.keys():
        if re_net.match(k):
            mac[k] = parse_mac(vm[k])
        elif re_dev.match(k):
            devices[k] = parse_dev(vm[k])
    results['mac'] = mac
    results['devices'] = devices
    results['vmid'] = int(vmid)