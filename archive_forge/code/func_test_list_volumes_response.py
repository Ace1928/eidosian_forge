from libcloud.pricing import get_pricing
from libcloud.compute.base import Node, NodeImage, NodeLocation, StorageVolume
def test_list_volumes_response(self):
    if not self.should_list_volumes:
        return None
    volumes = self.driver.list_volumes()
    self.assertTrue(isinstance(volumes, list))
    for volume in volumes:
        self.assertTrue(isinstance(volume, StorageVolume))