import boto
from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.address import Address
from boto.ec2.blockdevicemapping import BlockDeviceMapping
from boto.ec2.image import ProductCodes
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.group import Group
import base64
def modify_attribute(self, attribute, value, dry_run=False):
    """
        Changes an attribute of this instance

        :type attribute: string
        :param attribute: The attribute you wish to change.

            * instanceType - A valid instance type (m1.small)
            * kernel - Kernel ID (None)
            * ramdisk - Ramdisk ID (None)
            * userData - Base64 encoded String (None)
            * disableApiTermination - Boolean (true)
            * instanceInitiatedShutdownBehavior - stop|terminate
            * sourceDestCheck - Boolean (true)
            * groupSet - Set of Security Groups or IDs
            * ebsOptimized - Boolean (false)

        :type value: string
        :param value: The new value for the attribute

        :rtype: bool
        :return: Whether the operation succeeded or not
        """
    return self.connection.modify_instance_attribute(self.id, attribute, value, dry_run=dry_run)