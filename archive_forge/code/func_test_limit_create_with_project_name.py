import os
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_limit_create_with_project_name(self):
    registered_limit_id = self._create_dummy_registered_limit()
    raw_output = self.openstack('registered limit show %s' % registered_limit_id, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    service_id = self._extract_value_from_items('service_id', items)
    resource_name = self._extract_value_from_items('resource_name', items)
    raw_output = self.openstack('service show %s' % service_id)
    items = self.parse_show(raw_output)
    service_name = self._extract_value_from_items('name', items)
    project_name = self._create_dummy_project()
    params = {'project_name': project_name, 'service_name': service_name, 'resource_name': resource_name, 'resource_limit': 15}
    raw_output = self.openstack('limit create --project %(project_name)s --service %(service_name)s --resource-limit %(resource_limit)s %(resource_name)s' % params, cloud=SYSTEM_CLOUD)
    items = self.parse_show(raw_output)
    limit_id = self._extract_value_from_items('id', items)
    self.addCleanup(self.openstack, 'limit delete %s' % limit_id, cloud=SYSTEM_CLOUD)
    self.assert_show_fields(items, self.LIMIT_FIELDS)
    registered_limit_id = self._create_dummy_registered_limit()