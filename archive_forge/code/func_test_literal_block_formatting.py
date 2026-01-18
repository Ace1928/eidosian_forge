import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def test_literal_block_formatting(self):
    description = "Here's another description.\n\n    This one has a literal block.\n    These lines should be kept apart.\n    They should not be wrapped, even though they may be longer than 70 chars\n"
    expected = '# Here\'s another description.\n#\n#     This one has a literal block.\n#     These lines should be kept apart.\n#     They should not be wrapped, even though they may be longer than 70 chars\n#"admin": "is_admin:True"\n\n'
    self._test_formatting(description, expected)