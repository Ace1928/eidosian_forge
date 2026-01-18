import argparse
import collections
import io
import json
import logging
import os
import sys
from xml.etree import ElementTree as ET
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang import parse
def get_json(self):
    """Return a JSON representation of the test specification."""
    return json.dumps([spec.as_odict() for _, spec in sorted(self.tests.items())], indent=2)