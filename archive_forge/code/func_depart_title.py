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
def depart_title(self, node):
    if isinstance(node.parent, docutils.nodes.section) or isinstance(node.parent, docutils.nodes.document):
        self.set_to_parent()