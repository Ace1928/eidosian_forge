from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_otptoken_dict(ansible_to_ipa, uniqueid=None, newuniqueid=None, otptype=None, secretkey=None, description=None, owner=None, enabled=None, notbefore=None, notafter=None, vendor=None, model=None, serial=None, algorithm=None, digits=None, offset=None, interval=None, counter=None):
    """Create the dictionary of settings passed in"""
    otptoken = {}
    if uniqueid is not None:
        otptoken[ansible_to_ipa['uniqueid']] = uniqueid
    if newuniqueid is not None:
        otptoken[ansible_to_ipa['newuniqueid']] = newuniqueid
    if otptype is not None:
        otptoken[ansible_to_ipa['otptype']] = otptype.upper()
    if secretkey is not None:
        otptoken[ansible_to_ipa['secretkey']] = base64_to_base32(secretkey)
    if description is not None:
        otptoken[ansible_to_ipa['description']] = description
    if owner is not None:
        otptoken[ansible_to_ipa['owner']] = owner
    if enabled is not None:
        otptoken[ansible_to_ipa['enabled']] = False if enabled else True
    if notbefore is not None:
        otptoken[ansible_to_ipa['notbefore']] = notbefore + 'Z'
    if notafter is not None:
        otptoken[ansible_to_ipa['notafter']] = notafter + 'Z'
    if vendor is not None:
        otptoken[ansible_to_ipa['vendor']] = vendor
    if model is not None:
        otptoken[ansible_to_ipa['model']] = model
    if serial is not None:
        otptoken[ansible_to_ipa['serial']] = serial
    if algorithm is not None:
        otptoken[ansible_to_ipa['algorithm']] = algorithm
    if digits is not None:
        otptoken[ansible_to_ipa['digits']] = str(digits)
    if offset is not None:
        otptoken[ansible_to_ipa['offset']] = str(offset)
    if interval is not None:
        otptoken[ansible_to_ipa['interval']] = str(interval)
    if counter is not None:
        otptoken[ansible_to_ipa['counter']] = str(counter)
    return otptoken