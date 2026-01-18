import unittest
import time
import telnetlib
import socket
from nose.plugins.attrib import attr
from boto.ec2.connection import EC2Connection
from boto.exception import EC2ResponseError
import boto.ec2
def test_dry_run(self):
    c = EC2Connection()
    dry_run_msg = 'Request would have succeeded, but DryRun flag is set.'
    try:
        rs = c.get_all_images(dry_run=True)
        self.fail('Should have gotten an exception')
    except EC2ResponseError as e:
        self.assertTrue(dry_run_msg in str(e))
    try:
        rs = c.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', dry_run=True)
        self.fail('Should have gotten an exception')
    except EC2ResponseError as e:
        self.assertTrue(dry_run_msg in str(e))
    rs = c.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small')
    time.sleep(120)
    try:
        rs = c.stop_instances(instance_ids=[rs.instances[0].id], dry_run=True)
        self.fail('Should have gotten an exception')
    except EC2ResponseError as e:
        self.assertTrue(dry_run_msg in str(e))
    try:
        rs = c.terminate_instances(instance_ids=[rs.instances[0].id], dry_run=True)
        self.fail('Should have gotten an exception')
    except EC2ResponseError as e:
        self.assertTrue(dry_run_msg in str(e))
    rs.instances[0].terminate()