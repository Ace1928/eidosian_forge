from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def run_api_async(self):
    """ calls the REST API """
    args = [self.rest_api, self.api]
    kwargs = {}
    if self.method.upper() == 'POST':
        method = rest_generic.post_async
        kwargs['body'] = self.body
        kwargs['files'] = self.files
    elif self.method.upper() == 'PATCH':
        method = rest_generic.patch_async
        args.append(None)
        kwargs['body'] = self.body
        kwargs['files'] = self.files
    elif self.method.upper() == 'DELETE':
        method = rest_generic.delete_async
        args.append(None)
    else:
        self.module.warn('wait_for_completion ignored for %s method.' % self.method)
        return self.run_api()
    kwargs.update({'raw_error': True, 'headers': self.build_headers()})
    if self.query:
        kwargs['query'] = self.query
    response, error = method(*args, **kwargs)
    self.fail_on_error(0, response, error)
    return (0, response)