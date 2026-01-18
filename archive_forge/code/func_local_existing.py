from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def local_existing(gexisting):
    jp_bidir = False
    isauth = False
    if gexisting:
        jp_bidir = gexisting.get('jp_bidir')
        isauth = gexisting.get('isauth')
        if jp_bidir and isauth:
            gexisting.pop('jp_bidir')
            gexisting.pop('isauth')
    return (gexisting, jp_bidir, isauth)