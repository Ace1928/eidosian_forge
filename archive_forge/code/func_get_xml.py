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
def get_xml(self):
    """Return an XML representation of the test specification."""
    root = ET.Element('ctest')
    for _, spec in sorted(self.tests.items()):
        root.append(spec.as_element())
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf)
    return buf.getvalue().decode('utf-8')