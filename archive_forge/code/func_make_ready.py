from __future__ import print_function
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, ListProperty, ReferenceProperty, CalculatedProperty
from boto.manage.server import Server
from boto.manage import propget
import boto.utils
import boto.ec2
import time
import traceback
from contextlib import closing
import datetime
def make_ready(self, server):
    self.server = server
    self.put()
    self.install_xfs()
    self.attach()
    self.wait()
    self.format()
    self.mount()