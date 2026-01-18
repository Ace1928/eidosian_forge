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
def append_pending_ids(self, el):
    if self.settings.create_links:
        for id in self.pending_ids:
            SubElement(el, 'text:reference-mark', attrib={'text:name': id})
    self.pending_ids = []