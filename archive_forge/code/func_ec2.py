import boto
import boto.utils
from boto.compat import StringIO
from boto.mashups.iobject import IObject
from boto.pyami.config import Config, BotoConfigPath
from boto.mashups.interactive import interactive_shell
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty
import os
@property
def ec2(self):
    if self._ec2 is None:
        self._ec2 = boto.connect_ec2()
    return self._ec2