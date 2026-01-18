import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
def parse_show_as_object(self, raw_output):
    """Return a dict with values parsed from cli output."""
    items = self.parse_show(raw_output)
    o = {}
    for item in items:
        o.update(item)
    return o