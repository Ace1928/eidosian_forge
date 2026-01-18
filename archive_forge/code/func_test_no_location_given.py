import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
def test_no_location_given(self):
    self.assertIsNone(self.pp.get_price('1xCPU-1GB'))