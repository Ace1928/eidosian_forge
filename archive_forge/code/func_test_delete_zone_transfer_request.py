import unittest
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.client import DesignateCLI
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TransferRequestFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_delete_zone_transfer_request(self):
    transfer_request = self.useFixture(TransferRequestFixture(zone=self.zone, user='default', target_user='alt')).transfer_request
    self.clients.zone_transfer_request_delete(transfer_request.id)
    self.assertRaises(CommandFailed, self.clients.zone_transfer_request_show, transfer_request.id)