import argparse
import openstack.config
from openstack.tests.unit.config import base
def test_get_cloud_region_with_arg_parser(self):
    cloud_region = openstack.config.get_cloud_region(options=argparse.ArgumentParser(), validate=False)
    self.assertIsInstance(cloud_region, openstack.config.cloud_region.CloudRegion)