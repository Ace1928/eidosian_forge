from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def container_create(self, image='cirros', name=None, params=''):
    """Create container and add cleanup.

        :param String image: Image for a new container
        :param String name: Name for a new container
        :param String params: Additional args and kwargs
        :return: JSON object of created container
        """
    if not name:
        name = data_utils.rand_name('container')
    opts = self.get_opts()
    output = self.openstack('appcontainer create {0} --name {1} {2} {3}'.format(opts, name, image, params))
    container = jsonutils.loads(output)
    if not output:
        self.fail('Container has not been created!')
    return container