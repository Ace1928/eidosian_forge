from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_239(self):
    inp = "\n        first_name: Art\n        occupation: Architect\n        # I'm safe\n        about: Art Vandelay is a fictional character that George invents...\n        # we are not :(\n        # help me!\n        ---\n        # what?!\n        hello: world\n        # someone call the Batman\n        foo: bar # or quz\n        # Lost again\n        ---\n        I: knew\n        # final words\n        "
    d = YAML().round_trip_all(inp)