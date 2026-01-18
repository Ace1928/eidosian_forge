from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def fcmap_probe(self, data):
    props = {}
    props_not_supported = []
    if self.source:
        if data['source_vdisk_name'] != self.source:
            props_not_supported.append('source')
    if self.target:
        if data['target_vdisk_name'] != self.target:
            props_not_supported.append('target')
    if self.copytype:
        if self.copytype == 'snapshot' and data['autodelete'] == 'on' or (self.copytype == 'clone' and data['autodelete'] != 'on'):
            props_not_supported.append('copytype')
    if self.grainsize:
        if data['grain_size'] != self.grainsize:
            props_not_supported.append('grainsize')
    if props_not_supported:
        self.module.fail_json(msg='Update not supported for parameter: ' + ', '.join(props_not_supported))
    self.log('Probe which properties need to be updated...')
    if data['group_name'] and self.noconsistgrp:
        props['consistgrp'] = 0
    if not self.noconsistgrp:
        if self.consistgrp:
            if self.consistgrp != data['group_name']:
                props['consistgrp'] = self.consistgrp
    if self.copyrate:
        if self.copyrate != data['copy_rate']:
            props['copyrate'] = self.copyrate
    return props