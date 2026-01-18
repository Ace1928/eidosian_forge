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
def depart_topic(self, node):
    if 'classes' in node.attributes:
        if 'contents' in node.attributes['classes']:
            if self.settings.generate_oowriter_toc:
                self.update_toc_page_numbers(self.table_of_content_index_body)
                self.set_current_element(self.save_current_element)
            else:
                self.append_p('horizontalline')
            self.in_table_of_contents = False