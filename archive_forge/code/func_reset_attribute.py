import boto
from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.address import Address
from boto.ec2.blockdevicemapping import BlockDeviceMapping
from boto.ec2.image import ProductCodes
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.group import Group
import base64
def reset_attribute(self, attribute, dry_run=False):
    """
        Resets an attribute of this instance to its default value.

        :type attribute: string
        :param attribute: The attribute to reset. Valid values are:
                          kernel|ramdisk

        :rtype: bool
        :return: Whether the operation succeeded or not
        """
    return self.connection.reset_instance_attribute(self.id, attribute, dry_run=dry_run)