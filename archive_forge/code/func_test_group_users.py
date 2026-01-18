import boto
import time
from tests.compat import unittest
def test_group_users(self):
    iam = boto.connect_iam()
    name = 'boto-test-%d' % time.time()
    username = 'boto-test-user-%d' % time.time()
    iam.create_group(name)
    iam.create_user(username)
    iam.add_user_to_group(name, username)
    iam.remove_user_from_group(name, username)
    iam.delete_user(username)
    iam.delete_group(name)