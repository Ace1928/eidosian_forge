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
def checkfs(self, use_cmd=None):
    if self.server is None:
        raise ValueError('server attribute must be set to run this command')
    if use_cmd:
        cmd = use_cmd
    else:
        cmd = self.server.get_cmdshell()
    status = cmd.run('xfs_check %s' % self.device)
    if not use_cmd:
        cmd.close()
    if status[1].startswith('bad superblock magic number 0'):
        return False
    return True