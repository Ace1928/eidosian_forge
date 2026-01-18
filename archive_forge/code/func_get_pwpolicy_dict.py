from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_pwpolicy_dict(maxpwdlife=None, minpwdlife=None, historylength=None, minclasses=None, minlength=None, priority=None, maxfailcount=None, failinterval=None, lockouttime=None, gracelimit=None, maxrepeat=None, maxsequence=None, dictcheck=None, usercheck=None):
    pwpolicy = {}
    pwpolicy_options = {'krbmaxpwdlife': maxpwdlife, 'krbminpwdlife': minpwdlife, 'krbpwdhistorylength': historylength, 'krbpwdmindiffchars': minclasses, 'krbpwdminlength': minlength, 'cospriority': priority, 'krbpwdmaxfailure': maxfailcount, 'krbpwdfailurecountinterval': failinterval, 'krbpwdlockoutduration': lockouttime, 'passwordgracelimit': gracelimit, 'ipapwdmaxrepeat': maxrepeat, 'ipapwdmaxsequence': maxsequence}
    pwpolicy_boolean_options = {'ipapwddictcheck': dictcheck, 'ipapwdusercheck': usercheck}
    for option, value in pwpolicy_options.items():
        if value is not None:
            pwpolicy[option] = to_native(value)
    for option, value in pwpolicy_boolean_options.items():
        if value is not None:
            pwpolicy[option] = bool(value)
    return pwpolicy