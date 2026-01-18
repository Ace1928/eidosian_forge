import boto
from tests.compat import unittest
from boto.cloudhsm.exceptions import InvalidRequestException
def test_hapgs(self):
    label = 'my-hapg'
    response = self.cloudhsm.create_hapg(label=label)
    hapg_arn = response['HapgArn']
    self.addCleanup(self.cloudhsm.delete_hapg, hapg_arn)
    response = self.cloudhsm.list_hapgs()
    self.assertIn(hapg_arn, response['HapgList'])