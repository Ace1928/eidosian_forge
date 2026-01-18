from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils.basic import AnsibleModule
def find_customer_id(self, customer):
    for i in customer['customers']:
        if self.customer == i['customer'] and self.match == i['match']:
            return i['id']
    return None