from openstack.tests.functional import base
def test_download_object(self):
    result = self.conn.object_store.download_object(self.FILE, container=self.FOLDER)
    self.assertEqual(self.DATA, result)
    result = self.conn.object_store.download_object(self.sot)
    self.assertEqual(self.DATA, result)