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
def depart_citation(self, node):
    self.citation_id = None
    self.paragraph_style_stack.pop()
    self.bumped_list_level_stack.pop()
    self.in_citation = False