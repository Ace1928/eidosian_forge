from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def augment_subset(self, subset, info):
    if subset == 'lun_info' and info:
        for lun_info in info.values():
            serial = lun_info.get('serial_number') or lun_info.get('serial-number')
            if serial:
                hexlify = codecs.getencoder('hex')
                lun_info['serial_hex'] = to_text(hexlify(to_bytes(lun_info['serial_number']))[0])
                lun_info['naa_id'] = 'naa.600a0980' + lun_info['serial_hex']
    return info