import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def test_associate_elastic_ip(self):
    interface = NetworkInterfaceSpecification(associate_public_ip_address=False, subnet_id=self.subnet.id, delete_on_termination=True)
    interfaces = NetworkInterfaceCollection(interface)
    reservation = self.api.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', network_interfaces=interfaces)
    instance = reservation.instances[0]
    self.instances.append(instance)
    self.addCleanup(self.terminate_instances)
    igw = self.api.create_internet_gateway()
    time.sleep(5)
    self.api.attach_internet_gateway(igw.id, self.vpc.id)
    self.post_terminate_cleanups.append((self.api.detach_internet_gateway, (igw.id, self.vpc.id)))
    self.post_terminate_cleanups.append((self.api.delete_internet_gateway, (igw.id,)))
    eip = self.api.allocate_address('vpc')
    self.post_terminate_cleanups.append((self.delete_elastic_ip, (eip,)))
    time.sleep(60)
    eip.associate(instance.id)