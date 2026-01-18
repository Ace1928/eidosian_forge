from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_sudorule_dict(cmdcategory=None, description=None, hostcategory=None, ipaenabledflag=None, usercategory=None, runasgroupcategory=None, runasusercategory=None):
    data = {}
    if cmdcategory is not None:
        data['cmdcategory'] = cmdcategory
    if description is not None:
        data['description'] = description
    if hostcategory is not None:
        data['hostcategory'] = hostcategory
    if ipaenabledflag is not None:
        data['ipaenabledflag'] = ipaenabledflag
    if usercategory is not None:
        data['usercategory'] = usercategory
    if runasusercategory is not None:
        data['ipasudorunasusercategory'] = runasusercategory
    if runasgroupcategory is not None:
        data['ipasudorunasgroupcategory'] = runasgroupcategory
    return data