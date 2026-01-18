from libcloud.pricing import get_pricing
from libcloud.compute.base import Node, NodeImage, NodeLocation, StorageVolume
def test_get_pricing_success(self):
    if not self.should_have_pricing:
        return None
    driver_type = 'compute'
    try:
        get_pricing(driver_type=driver_type, driver_name=self.driver.api_name)
    except KeyError:
        self.fail('No {driver_type!r} pricing info for {driver}.'.format(driver=self.driver.__class__.__name__, driver_type=driver_type))