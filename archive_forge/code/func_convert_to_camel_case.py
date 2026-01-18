from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
from re import sub
def convert_to_camel_case(string):
    string = sub('(_|-)+', ' ', string).title().replace(' ', '')
    return string[0].lower() + string[1:]