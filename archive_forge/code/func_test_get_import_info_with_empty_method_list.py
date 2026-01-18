import glance.api.v2.discovery
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_import_info_with_empty_method_list(self):
    """When methods list is empty, should still return import methods"""
    self.config(enabled_import_methods=[])
    req = unit_test_utils.get_fake_request()
    output = self.controller.get_image_import(req)
    self.assertIn('import-methods', output)
    self.assertEqual([], output['import-methods']['value'])