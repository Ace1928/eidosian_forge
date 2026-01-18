import glance.api.v2.discovery
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_import_info(self):
    """Testing defaults, not all possible values"""
    default_import_methods = ['glance-direct', 'web-download', 'copy-image']
    req = unit_test_utils.get_fake_request()
    output = self.controller.get_image_import(req)
    self.assertIn('import-methods', output)
    self.assertEqual(default_import_methods, output['import-methods']['value'])