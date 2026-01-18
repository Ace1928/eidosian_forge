import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
def header_section(self):
    """Returns an ARFF header as a string."""
    s = '% Weka ARFF file\n' + '% Generated automatically by NLTK\n' + '%% %s\n\n' % time.ctime()
    s += '@RELATION rel\n\n'
    for fname, ftype in self._features:
        s += '@ATTRIBUTE %-30r %s\n' % (fname, ftype)
    s += '@ATTRIBUTE %-30r {%s}\n' % ('-label-', ','.join(self._labels))
    return s