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
def store_embedded_files(self, zfile):
    embedded_files = self.visitor.get_embedded_file_list()
    for source, destination in embedded_files:
        if source is None:
            continue
        try:
            zfile.write(source, destination)
        except OSError:
            self.document.reporter.warning("Can't open file %s." % (source,))