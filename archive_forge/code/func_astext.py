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
def astext(self):
    root = self.content_tree.getroot()
    et = etree.ElementTree(root)
    s1 = ToString(et)
    return s1