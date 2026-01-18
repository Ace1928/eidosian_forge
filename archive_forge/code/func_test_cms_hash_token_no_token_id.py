import errno
import os
import subprocess
from unittest import mock
import testresources
from testtools import matchers
from keystoneclient.common import cms
from keystoneclient import exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
def test_cms_hash_token_no_token_id(self):
    token_id = None
    self.assertThat(cms.cms_hash_token(token_id), matchers.Is(None))