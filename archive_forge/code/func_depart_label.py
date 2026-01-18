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
def depart_label(self, node):
    if isinstance(node.parent, docutils.nodes.footnote):
        pass
    elif self.citation_id is not None:
        if self.settings.create_links:
            self.append_child('text:reference-mark-end', attrib={'text:name': '%s' % (self.citation_id,)})
            el0 = SubElement(self.current_element, 'text:span')
            el0.text = ']'
        else:
            self.current_element.text += ']'
        self.set_to_parent()