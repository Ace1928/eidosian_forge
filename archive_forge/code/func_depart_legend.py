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
def depart_legend(self, node):
    if isinstance(node.parent, docutils.nodes.figure):
        self.paragraph_style_stack.pop()
        self.set_to_parent()
        self.set_to_parent()
        self.set_to_parent()