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
def test_invalid_formatting(self):
    description = "Here's a broken description.\n\nWe have some text...\n    Followed by a literal block without any spaces.\n    We don't support definition lists, so this is just wrong!\n"
    expected = '# Here\'s a broken description.\n#\n# We have some text...\n#\n#     Followed by a literal block without any spaces.\n#     We don\'t support definition lists, so this is just wrong!\n#"admin": "is_admin:True"\n\n'
    with warnings.catch_warnings(record=True) as warns:
        self._test_formatting(description, expected)
        self.assertEqual(1, len(warns))
        self.assertTrue(issubclass(warns[-1].category, FutureWarning))
        self.assertIn('Invalid policy description', str(warns[-1].message))