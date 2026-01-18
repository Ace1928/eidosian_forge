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
def append_p(self, style, text=None):
    result = self.append_child('text:p', attrib={'text:style-name': self.rststyle(style)})
    self.append_pending_ids(result)
    if text is not None:
        result.text = text
    return result