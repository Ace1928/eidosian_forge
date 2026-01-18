from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def find_fs_id(module, system, fs_name):
    """ Find the ID of this fs """
    fs_url = f'filesystems?name={fs_name}&fields=id'
    fs = system.api.get(path=fs_url)
    result = fs.get_json()['result']
    if len(result) != 1:
        module.fail_json(f"Cannot find a file ststem with name '{fs_name}'")
    fs_id = result[0]['id']
    return fs_id