import boto.ec2
from boto.mashups.iobject import IObject
from boto.pyami.config import BotoConfigPath, Config
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty, CalculatedProperty
from boto.manage import propget
from boto.ec2.zone import Zone
from boto.ec2.keypair import KeyPair
import os, time
from contextlib import closing
from boto.exception import EC2ResponseError
from boto.compat import six, StringIO
def bundle_image(self, prefix, size, ssh_key):
    command = ''
    if self.uname != 'root':
        command = 'sudo '
    command += 'ec2-bundle-vol '
    command += '-c %s -k %s ' % (self.remote_cert_file, self.remote_key_file)
    command += '-u %s ' % self.server._reservation.owner_id
    command += '-p %s ' % prefix
    command += '-s %d ' % size
    command += '-d /mnt '
    if self.server.instance_type == 'm1.small' or self.server.instance_type == 'c1.medium':
        command += '-r i386'
    else:
        command += '-r x86_64'
    return command