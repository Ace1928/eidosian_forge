from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.intersight.plugins.module_utils.intersight import IntersightModule, intersight_argument_spec
def check_and_add_prop(prop, propKey, params, api_body):
    if propKey in params.keys():
        api_body[prop] = params[propKey]