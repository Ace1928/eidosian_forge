import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def prompt_for_password(prompt=None):
    """Fake prompt function that just returns a constant string"""
    return 'promptpass'