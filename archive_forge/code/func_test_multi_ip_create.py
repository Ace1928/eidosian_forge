import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def test_multi_ip_create(self):
    interface = NetworkInterfaceSpecification(device_index=0, subnet_id=self.subnet.id, private_ip_address='10.0.0.21', description='This is a test interface using boto.', delete_on_termination=True, private_ip_addresses=[PrivateIPAddress(private_ip_address='10.0.0.22', primary=False), PrivateIPAddress(private_ip_address='10.0.0.23', primary=False), PrivateIPAddress(private_ip_address='10.0.0.24', primary=False)])
    interfaces = NetworkInterfaceCollection(interface)
    reservation = self.api.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', network_interfaces=interfaces)
    time.sleep(10)
    instance = reservation.instances[0]
    self.addCleanup(self.terminate_instance, instance)
    retrieved = self.api.get_all_reservations(instance_ids=[instance.id])
    self.assertEqual(len(retrieved), 1)
    retrieved_instances = retrieved[0].instances
    self.assertEqual(len(retrieved_instances), 1)
    retrieved_instance = retrieved_instances[0]
    self.assertEqual(len(retrieved_instance.interfaces), 1)
    interface = retrieved_instance.interfaces[0]
    private_ip_addresses = interface.private_ip_addresses
    self.assertEqual(len(private_ip_addresses), 4)
    self.assertEqual(private_ip_addresses[0].private_ip_address, '10.0.0.21')
    self.assertEqual(private_ip_addresses[0].primary, True)
    self.assertEqual(private_ip_addresses[1].private_ip_address, '10.0.0.22')
    self.assertEqual(private_ip_addresses[2].private_ip_address, '10.0.0.23')
    self.assertEqual(private_ip_addresses[3].private_ip_address, '10.0.0.24')