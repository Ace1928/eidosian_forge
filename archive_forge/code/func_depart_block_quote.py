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
def depart_block_quote(self, node):
    self.paragraph_style_stack.pop()
    self.blockstyle = ''
    self.line_indent_level -= 1