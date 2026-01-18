from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def addvdiskcopy(self):
    self.log('Entering function addvdiskcopy')
    cmd = 'addvdiskcopy'
    cmdopts = {}
    if self.size:
        self.module.fail_json(msg="Parameter 'size' cannot be passed while converting a standard volume to Mirror Volume")
    siteA, siteB = self.discover_site_from_pools()
    if siteA != siteB:
        self.module.fail_json(msg='To create Standard Mirrored volume, provide pools belonging to same site.')
    if self.poolA and (self.poolB == self.discovered_standard_vol_pool and self.poolA != self.discovered_standard_vol_pool):
        cmdopts['mdiskgrp'] = self.poolA
    elif self.poolB and (self.poolA == self.discovered_standard_vol_pool and self.poolB != self.discovered_standard_vol_pool):
        cmdopts['mdiskgrp'] = self.poolB
    else:
        self.module.fail_json(msg='One of the input pools must belong to the volume')
    if self.compressed:
        cmdopts['compressed'] = self.compressed
    if self.grainsize:
        cmdopts['grainsize'] = self.grainsize
    if self.thin and self.rsize:
        cmdopts['rsize'] = self.rsize
    elif self.thin:
        cmdopts['rsize'] = '2%'
    elif self.rsize and (not self.thin):
        self.module.fail_json(msg="To configure 'rsize', parameter 'thin' should be passed and the value should be 'true'.")
    if self.deduplicated:
        if self.thin:
            cmdopts['deduplicated'] = self.deduplicated
            cmdopts['autoexpand'] = True
        else:
            self.module.fail_json(msg="To configure 'deduplicated', parameter 'thin' should be passed and the value should be 'true.'")
    if self.isdrp and self.thin:
        cmdopts['autoexpand'] = True
    if self.module.check_mode:
        self.changed = True
        return
    cmdargs = [self.name]
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)