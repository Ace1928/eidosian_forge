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
def append_child(self, tag, attrib=None, parent=None):
    if parent is None:
        parent = self.current_element
    el = SubElement(parent, tag, attrib)
    return el