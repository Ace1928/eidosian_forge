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
def attach_page_style(self, el):
    """Attach the default page style.

        Create an automatic-style that refers to the current style
        of this element and that refers to the default page style.
        """
    current_style = el.get('text:style-name')
    style_name = 'P1003'
    el1 = SubElement(self.automatic_styles, 'style:style', attrib={'style:name': style_name, 'style:master-page-name': 'rststyle-pagedefault', 'style:family': 'paragraph'}, nsdict=SNSD)
    if current_style:
        el1.set('style:parent-style-name', current_style)
    el.set('text:style-name', style_name)