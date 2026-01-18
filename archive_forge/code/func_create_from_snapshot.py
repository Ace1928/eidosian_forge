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
def create_from_snapshot(self, name, snapshot, size=None):
    if size < self.size:
        size = self.size
    ec2 = self.get_ec2_connection()
    if self.zone_name is None or self.zone_name == '':
        current_volume = ec2.get_all_volumes([self.volume_id])[0]
        self.zone_name = current_volume.zone
    ebs_volume = ec2.create_volume(size, self.zone_name, snapshot)
    v = Volume()
    v.ec2 = self.ec2
    v.volume_id = ebs_volume.id
    v.name = name
    v.mount_point = self.mount_point
    v.device = self.device
    v.region_name = self.region_name
    v.zone_name = self.zone_name
    v.put()
    return v