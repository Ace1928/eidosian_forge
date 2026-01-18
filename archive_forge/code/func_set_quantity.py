import boto
import boto.ec2
from boto.mashups.server import Server, ServerSet
from boto.mashups.iobject import IObject
from boto.pyami.config import Config
from boto.sdb.persist import get_domain, set_domain
import time
from boto.compat import StringIO
def set_quantity(self, n=0):
    if n > 0:
        self.quantity = n
    else:
        self.quantity = self.get_int('Quantity')