from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.acme.acme import (
from ansible_collections.community.crypto.plugins.module_utils.acme.account import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import ModuleFailException
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def get_orders_list(module, client, orders_url):
    """
    Retrieves orders list (handles pagination).
    """
    orders = []
    while orders_url:
        res, info = client.get_request(orders_url, parse_json_result=True, fail_on_error=True)
        if not res.get('orders'):
            if orders:
                module.warn('When retrieving orders list part {0}, got empty result list'.format(orders_url))
            break
        orders.extend(res['orders'])
        new_orders_url = []

        def f(link, relation):
            if relation == 'next':
                new_orders_url.append(link)
        process_links(info, f)
        new_orders_url.append(None)
        previous_orders_url, orders_url = (orders_url, new_orders_url.pop(0))
        if orders_url == previous_orders_url:
            orders_url = None
    return orders