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
def depart_bullet_list(self, node):
    if self.in_table_of_contents:
        if self.settings.generate_oowriter_toc:
            pass
        else:
            self.set_to_parent()
            self.list_style_stack.pop()
    else:
        self.set_to_parent()
        self.list_style_stack.pop()
    self.list_level -= 1