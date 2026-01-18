from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_host_dict(description=None, force=None, ip_address=None, ns_host_location=None, ns_hardware_platform=None, ns_os_version=None, user_certificate=None, mac_address=None, random_password=None):
    data = {}
    if description is not None:
        data['description'] = description
    if force is not None:
        data['force'] = force
    if ip_address is not None:
        data['ip_address'] = ip_address
    if ns_host_location is not None:
        data['nshostlocation'] = ns_host_location
    if ns_hardware_platform is not None:
        data['nshardwareplatform'] = ns_hardware_platform
    if ns_os_version is not None:
        data['nsosversion'] = ns_os_version
    if user_certificate is not None:
        data['usercertificate'] = [{'__base64__': item} for item in user_certificate]
    if mac_address is not None:
        data['macaddress'] = mac_address
    if random_password is not None:
        data['random'] = random_password
    return data