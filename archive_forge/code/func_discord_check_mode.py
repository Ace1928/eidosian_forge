from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import AnsibleModule
def discord_check_mode(module):
    webhook_id = module.params['webhook_id']
    webhook_token = module.params['webhook_token']
    headers = {'content-type': 'application/json'}
    url = 'https://discord.com/api/webhooks/%s/%s' % (webhook_id, webhook_token)
    response, info = fetch_url(module, url, method='GET', headers=headers)
    return (response, info)