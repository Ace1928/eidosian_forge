from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def set_bit_mask(self, systemmask=None):
    cmd = 'chmdiskgrp'
    bit_mask = '1'.ljust(int(self.partnership_index) + 1, '0') if not systemmask else systemmask
    cmdopts = {'replicationpoollinkedsystemsmask': bit_mask}
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=[self.name])