import boto
import boto.ec2
from boto.mashups.server import Server, ServerSet
from boto.mashups.iobject import IObject
from boto.pyami.config import Config
from boto.sdb.persist import get_domain, set_domain
import time
from boto.compat import StringIO
def set_zone(self, zone=None):
    if zone:
        self.zone = zone
    else:
        l = [(z, z.name, z.state) for z in self.ec2.get_all_zones()]
        self.zone = self.choose_from_list(l, prompt='Choose Availability Zone')