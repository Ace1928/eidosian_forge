import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def test_associate_public_ip(self):
    interface = NetworkInterfaceSpecification(associate_public_ip_address=True, subnet_id=self.subnet.id, delete_on_termination=True)
    interfaces = NetworkInterfaceCollection(interface)
    reservation = self.api.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', network_interfaces=interfaces)
    instance = reservation.instances[0]
    self.instances.append(instance)
    self.addCleanup(self.terminate_instances)
    time.sleep(60)
    retrieved = self.api.get_all_reservations(instance_ids=[instance.id])
    self.assertEqual(len(retrieved), 1)
    retrieved_instances = retrieved[0].instances
    self.assertEqual(len(retrieved_instances), 1)
    retrieved_instance = retrieved_instances[0]
    self.assertEqual(len(retrieved_instance.interfaces), 1)
    interface = retrieved_instance.interfaces[0]
    self.assertTrue(interface.publicIp.count('.') >= 3)