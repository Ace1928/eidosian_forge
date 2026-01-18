from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import (
from ansible.module_utils.basic import AnsibleModule
def info_strategy(api, wished_cn):
    cn_list = api.fetch_all_resources('namespaces')
    cn_lookup = dict(((fn['name'], fn) for fn in cn_list))
    if wished_cn['name'] not in cn_lookup:
        msg = "Error during container namespace lookup: Unable to find container namespace named '%s' in project '%s'" % (wished_cn['name'], wished_cn['project_id'])
        api.module.fail_json(msg=msg)
    target_cn = cn_lookup[wished_cn['name']]
    response = api.get(path=api.api_path + '/%s' % target_cn['id'])
    if not response.ok:
        msg = "Error during container namespace lookup: %s: '%s' (%s)" % (response.info['msg'], response.json['message'], response.json)
        api.module.fail_json(msg=msg)
    return response.json