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
def setup_page(self):
    self.setup_paper(self.dom_stylesheet)
    if len(self.header_content) > 0 or len(self.footer_content) > 0 or self.settings.custom_header or self.settings.custom_footer:
        self.add_header_footer(self.dom_stylesheet)
    new_content = etree.tostring(self.dom_stylesheet)
    return new_content