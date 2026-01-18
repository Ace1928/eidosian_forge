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
def rststyle(self, name, parameters=()):
    """
        Returns the style name to use for the given style.

        If `parameters` is given `name` must contain a matching number of
        ``%`` and is used as a format expression with `parameters` as
        the value.
        """
    name1 = name % parameters
    stylename = self.format_map.get(name1, 'rststyle-%s' % name1)
    return stylename