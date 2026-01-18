from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_user_dict(displayname=None, givenname=None, krbpasswordexpiration=None, loginshell=None, mail=None, nsaccountlock=False, sn=None, sshpubkey=None, telephonenumber=None, title=None, userpassword=None, gidnumber=None, uidnumber=None, homedirectory=None, userauthtype=None):
    user = {}
    if displayname is not None:
        user['displayname'] = displayname
    if krbpasswordexpiration is not None:
        user['krbpasswordexpiration'] = krbpasswordexpiration + 'Z'
    if givenname is not None:
        user['givenname'] = givenname
    if loginshell is not None:
        user['loginshell'] = loginshell
    if mail is not None:
        user['mail'] = mail
    user['nsaccountlock'] = nsaccountlock
    if sn is not None:
        user['sn'] = sn
    if sshpubkey is not None:
        user['ipasshpubkey'] = sshpubkey
    if telephonenumber is not None:
        user['telephonenumber'] = telephonenumber
    if title is not None:
        user['title'] = title
    if userpassword is not None:
        user['userpassword'] = userpassword
    if gidnumber is not None:
        user['gidnumber'] = gidnumber
    if uidnumber is not None:
        user['uidnumber'] = uidnumber
    if homedirectory is not None:
        user['homedirectory'] = homedirectory
    if userauthtype is not None:
        user['ipauserauthtype'] = userauthtype
    return user