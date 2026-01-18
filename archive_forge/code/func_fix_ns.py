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
def fix_ns(tag, attrib, nsdict):
    nstag = add_ns(tag, nsdict)
    nsattrib = {}
    for key, val in list(attrib.items()):
        nskey = add_ns(key, nsdict)
        nsattrib[nskey] = val
    return (nstag, nsattrib)