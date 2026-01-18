import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
def assert_show_fields(self, show_output, field_names):
    """Verify that all items have keys listed in field_names."""
    all_headers = [item for sublist in show_output for item in sublist]
    for field_name in field_names:
        self.assertIn(field_name, all_headers)