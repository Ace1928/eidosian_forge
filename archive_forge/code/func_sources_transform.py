from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_DTC_TOPOLOGY
from ..module_utils.api import normalize_ib_spec
def sources_transform(sources, module):
    source_list = list()
    for source in sources:
        src = dict([(k, v) for k, v in iteritems(source) if v is not None])
        if 'source_type' not in src or 'source_value' not in src:
            module.fail_json(msg='source_type and source_value are required for source')
        source_list.append(src)
    return source_list