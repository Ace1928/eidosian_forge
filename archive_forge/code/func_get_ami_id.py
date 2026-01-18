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
def get_ami_id(self, params):
    valid = False
    while not valid:
        ami = params.get('ami', None)
        if not ami:
            prop = StringProperty(name='ami', verbose_name='AMI')
            ami = propget.get(prop)
        try:
            rs = self.ec2.get_all_images([ami])
            if len(rs) == 1:
                valid = True
                params['ami'] = rs[0]
        except EC2ResponseError:
            pass