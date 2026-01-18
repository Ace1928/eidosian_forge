from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def get_partnership_detail(self, rest_obj, id):
    return rest_obj.svc_obj_info(cmd='lspartnership', cmdopts=None, cmdargs=[id])