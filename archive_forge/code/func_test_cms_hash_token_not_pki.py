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
def test_cms_hash_token_not_pki(self):
    """If the token_id is not a PKI token then it returns the token_id."""
    token = 'something'
    self.assertFalse(cms.is_asn1_token(token))
    self.assertThat(cms.cms_hash_token(token), matchers.Is(token))