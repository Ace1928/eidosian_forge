from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible.module_utils.basic import AnsibleModule
def get_customers(self):
    url = '%s/api/customers' % self.alerta_url
    response = self.send_request(url)
    pages = response['pages']
    if pages > 1:
        for page in range(2, pages + 1):
            page_url = url + '?page=' + str(page)
            new_results = self.send_request(page_url)
            response.update(new_results)
    return response