import boto
from boto.sdb.db.property import StringProperty, DateTimeProperty, IntegerProperty
from boto.sdb.db.model import Model
import datetime, subprocess, time
from boto.compat import StringIO
def check_hour(val):
    if val == '*':
        return
    if int(val) < 0 or int(val) > 23:
        raise ValueError