import unittest
import time
import telnetlib
import socket
from nose.plugins.attrib import attr
from boto.ec2.connection import EC2Connection
from boto.exception import EC2ResponseError
import boto.ec2
@attr('notdefault')
def test_launch_permissions(self):
    user_id = '963068290131'
    print('--- running EC2Connection tests ---')
    c = EC2Connection()
    rs = c.get_all_images(owners=[user_id])
    assert len(rs) > 0
    image = rs[0]
    status = image.set_launch_permissions(group_names=['all'])
    assert status
    d = image.get_launch_permissions()
    assert 'groups' in d
    assert len(d['groups']) > 0
    status = image.remove_launch_permissions(group_names=['all'])
    assert status
    time.sleep(10)
    d = image.get_launch_permissions()
    assert 'groups' not in d