from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def do_notify_bearychat(module, url, payload):
    response, info = fetch_url(module, url, data=payload)
    if info['status'] != 200:
        url_info = urlparse(url)
        obscured_incoming_webhook = urlunparse((url_info.scheme, url_info.netloc, '[obscured]', '', '', ''))
        module.fail_json(msg=' failed to send %s to %s: %s' % (payload, obscured_incoming_webhook, info['msg']))