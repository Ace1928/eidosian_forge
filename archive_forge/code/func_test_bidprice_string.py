from decimal import Decimal
from tests.compat import unittest
from boto.emr.instance_group import InstanceGroup
def test_bidprice_string(self):
    """
        Test InstanceGroup init works with bidprice type = string.
        """
    instance_group = InstanceGroup(1, 'MASTER', 'm1.small', 'SPOT', 'master', bidprice='1.1')
    self.assertEquals('1.1', instance_group.bidprice)