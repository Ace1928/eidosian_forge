import time
import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_encryption_type(self):
    name = uuid.uuid4().hex
    encryption_type = uuid.uuid4().hex
    cmd_output = self.openstack('volume type create --encryption-provider LuksEncryptor --encryption-cipher aes-xts-plain64 --encryption-key-size 128 --encryption-control-location front-end ' + encryption_type, parse_output=True)
    expected = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
    for attr, value in expected.items():
        self.assertEqual(value, cmd_output['encryption'][attr])
    cmd_output = self.openstack('volume type show --encryption-type ' + encryption_type, parse_output=True)
    expected = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
    for attr, value in expected.items():
        self.assertEqual(value, cmd_output['encryption'][attr])
    cmd_output = self.openstack('volume type list --encryption-type', parse_output=True)
    encryption_output = [t['Encryption'] for t in cmd_output if t['Name'] == encryption_type][0]
    expected = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
    for attr, value in expected.items():
        self.assertEqual(value, encryption_output[attr])
    raw_output = self.openstack('volume type set --encryption-key-size 256 --encryption-control-location back-end ' + encryption_type)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('volume type show --encryption-type ' + encryption_type, parse_output=True)
    expected = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 256, 'control_location': 'back-end'}
    for attr, value in expected.items():
        self.assertEqual(value, cmd_output['encryption'][attr])
    cmd_output = self.openstack('volume type create --private ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume type delete ' + name)
    self.assertEqual(name, cmd_output['name'])
    raw_output = self.openstack('volume type set --encryption-provider LuksEncryptor --encryption-cipher aes-xts-plain64 --encryption-key-size 128 --encryption-control-location front-end ' + name)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('volume type show --encryption-type ' + name, parse_output=True)
    expected = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
    for attr, value in expected.items():
        self.assertEqual(value, cmd_output['encryption'][attr])
    raw_output = self.openstack('volume type unset --encryption-type ' + name)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('volume type show --encryption-type ' + name, parse_output=True)
    self.assertEqual({}, cmd_output['encryption'])
    raw_output = self.openstack('volume type delete ' + encryption_type)
    self.assertEqual('', raw_output)