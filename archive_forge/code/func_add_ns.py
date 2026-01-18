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
def add_ns(tag, nsdict=CNSD):
    if WhichElementTree == 'lxml':
        nstag, name = tag.split(':')
        ns = nsdict.get(nstag)
        if ns is None:
            raise RuntimeError('Invalid namespace prefix: %s' % nstag)
        tag = '{%s}%s' % (ns, name)
    return tag