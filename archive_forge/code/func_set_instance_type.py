import boto
import boto.ec2
from boto.mashups.server import Server, ServerSet
from boto.mashups.iobject import IObject
from boto.pyami.config import Config
from boto.sdb.persist import get_domain, set_domain
import time
from boto.compat import StringIO
def set_instance_type(self, instance_type=None):
    if instance_type:
        self.instance_type = instance_type
    else:
        self.instance_type = self.choose_from_list(InstanceTypes, 'Instance Type')