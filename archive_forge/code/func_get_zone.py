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
def get_zone(self, params):
    if not params.get('zone', None):
        prop = StringProperty(name='zone', verbose_name='EC2 Availability Zone', choices=self.ec2.get_all_zones)
        params['zone'] = propget.get(prop)