import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def get_image_width_height(self, node, attr):
    size = None
    unit = None
    if attr in node.attributes:
        size = node.attributes[attr]
        size = size.strip()
        try:
            if size.endswith('%'):
                if attr == 'height':
                    raise ValueError('percentage not allowed for height')
                size = size.rstrip(' %')
                size = float(size) / 100.0
                unit = '%'
            else:
                size, unit = self.convert_to_cm(size)
        except ValueError as exp:
            self.document.reporter.warning('Invalid %s for image: "%s".  Error: "%s".' % (attr, node.attributes[attr], exp))
    return (size, unit)