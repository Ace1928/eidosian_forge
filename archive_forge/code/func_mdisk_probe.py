from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdisk_probe(self, data):
    props = []
    if self.encrypt:
        if self.encrypt != data['encrypt']:
            props += ['encrypt']
    if props is []:
        props = None
    self.log("mdisk_probe props='%s'", data)
    return props