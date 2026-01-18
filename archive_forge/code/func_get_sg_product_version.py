from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
def get_sg_product_version(self, api_root='grid'):
    method = 'GET'
    api = 'api/v3/%s/config/product-version' % api_root
    message, error = self.send_request(method, api, params={})
    if error:
        self.module.fail_json(msg=error)
    self.set_version(message)