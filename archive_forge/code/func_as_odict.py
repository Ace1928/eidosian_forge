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
def as_odict(self):
    """Return a dictionary representation of the specification, suitable for
       serialization to JSON.
    """
    props = collections.OrderedDict()
    for key, value in sorted(self.props.items()):
        if key == 'LABELS':
            value = value.split(';')
        if key == 'TIMEOUT':
            value = int(value)
        if key.startswith('_'):
            continue
        props[key.lower()] = value
    out = collections.OrderedDict([('name', self.name), ('argv', self.argv), ('cwd', self.cwd), ('props', props)])
    return out