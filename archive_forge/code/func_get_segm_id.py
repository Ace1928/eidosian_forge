import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
def get_segm_id(self, elt):
    for attr in elt.attrib:
        if attr.endswith('id'):
            return elt.get(attr)