import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
def test_find_policy_file(self):
    policy_file = '/etc/policy.json'
    self.useFixture(fixtures.MonkeyPatch('os.path.exists', lambda p: p == policy_file))
    self.conf([])
    self.assertIsNone(self.conf.find_file('foo.json'))
    self.assertEqual(policy_file, self.conf.find_file('policy.json'))