import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def deploy_template_create(self, name, params=''):
    """Create baremetal deploy template and add cleanup.

        :param String name: deploy template name
        :param String params: additional parameters
        :return: JSON object of created deploy template
        """
    opts = self.get_opts()
    template = self.openstack('baremetal deploy template create {0} {1} {2}'.format(opts, name, params))
    template = json.loads(template)
    if not template:
        self.fail('Baremetal deploy template has not been created!')
    self.addCleanup(self.deploy_template_delete, template['uuid'], True)
    return template