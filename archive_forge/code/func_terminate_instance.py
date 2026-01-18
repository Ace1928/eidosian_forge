import time
import boto
from boto.compat import six
from tests.compat import unittest
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
def terminate_instance(self, instance):
    instance.terminate()
    for i in six.moves.range(300):
        instance.update()
        if instance.state == 'terminated':
            time.sleep(30)
            return
        else:
            time.sleep(10)