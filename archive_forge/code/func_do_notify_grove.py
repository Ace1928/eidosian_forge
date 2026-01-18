from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def do_notify_grove(module, channel_token, service, message, url=None, icon_url=None):
    my_url = BASE_URL % (channel_token,)
    my_data = dict(service=service, message=message)
    if url is not None:
        my_data['url'] = url
    if icon_url is not None:
        my_data['icon_url'] = icon_url
    data = urlencode(my_data)
    response, info = fetch_url(module, my_url, data=data)
    if info['status'] != 200:
        module.fail_json(msg='failed to send notification: %s' % info['msg'])