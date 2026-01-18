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
def get_db(self):
    """Return the parse database: dictionary of statement names to parse
       functions. While cmake-format uses regular functions we map statement
       names to callbacks within this context.
    """
    return {'add_test': self.parse_add_test, 'subdirs': self.parse_subdirs, 'set_tests_properties': self.parse_set_tests_properties}