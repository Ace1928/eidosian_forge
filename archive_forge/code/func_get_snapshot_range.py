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
def get_snapshot_range(self, snaps, start_date=None, end_date=None):
    l = []
    for snap in snaps:
        if start_date and end_date:
            if snap.date >= start_date and snap.date <= end_date:
                l.append(snap)
        elif start_date:
            if snap.date >= start_date:
                l.append(snap)
        elif end_date:
            if snap.date <= end_date:
                l.append(snap)
        else:
            l.append(snap)
    return l