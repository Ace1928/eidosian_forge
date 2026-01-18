from openstack.tests.functional.shared_file_system import base
def test_export_locations(self):
    exs = self.user_cloud.shared_file_system.export_locations(self.SHARE_ID)
    self.assertGreater(len(list(exs)), 0)
    for ex in exs:
        for attribute in ('id', 'path', 'share_instance_id', 'updated_at', 'created_at'):
            self.assertTrue(hasattr(ex, attribute))
            self.assertIsInstance(getattr(ex, attribute), 'str')
        for attribute in ('is_preferred', 'is_admin'):
            self.assertTrue(hasattr(ex, attribute))
            self.assertIsInstance(getattr(ex, attribute), 'bool')