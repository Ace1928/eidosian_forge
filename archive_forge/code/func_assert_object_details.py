import configparser
import os
import time
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
def assert_object_details(self, expected, items):
    """Check presence of common object properties.

        :param expected: expected object properties
        :param items: object properties
        """
    for value in expected:
        self.assertIn(value, items)