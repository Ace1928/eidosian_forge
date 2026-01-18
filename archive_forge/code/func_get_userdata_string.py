import boto
import boto.ec2
from boto.mashups.server import Server, ServerSet
from boto.mashups.iobject import IObject
from boto.pyami.config import Config
from boto.sdb.persist import get_domain, set_domain
import time
from boto.compat import StringIO
def get_userdata_string(self):
    s = StringIO()
    self.config.write(s)
    return s.getvalue()