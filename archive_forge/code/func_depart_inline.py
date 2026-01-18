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
def depart_inline(self, node):
    count = self.inline_style_count_stack.pop()
    for x in range(count):
        self.set_to_parent()