import io
import logging
import os
import pprint
import sys
import typing as ty
import fixtures
from oslotest import base
import testtools.content
from openstack.tests import fixtures as os_fixtures
from openstack import utils
def assertSubdict(self, part, whole):
    missing_keys = []
    for key in part:
        if not whole[key] and part[key]:
            missing_keys.append(key)
    if missing_keys:
        self.fail('Keys %s are in %s but not in %s' % (missing_keys, part, whole))
    wrong_values = [(key, part[key], whole[key]) for key in part if part[key] != whole[key]]
    if wrong_values:
        self.fail('Mismatched values: %s' % ', '.join(('for %s got %s and %s' % tpl for tpl in wrong_values)))