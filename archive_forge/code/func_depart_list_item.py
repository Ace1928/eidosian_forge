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
def depart_list_item(self, node):
    if self.in_table_of_contents:
        if self.settings.generate_oowriter_toc:
            self.paragraph_style_stack.pop()
        else:
            self.set_to_parent()
    else:
        if len(self.bumped_list_level_stack) > 0:
            level_obj = self.bumped_list_level_stack[-1]
            if level_obj.get_sibling():
                level_obj.set_nested(True)
                for level_obj1 in self.bumped_list_level_stack:
                    for idx in range(level_obj1.get_level()):
                        self.set_to_parent()
                        self.set_to_parent()
        self.paragraph_style_stack.pop()
        self.set_to_parent()