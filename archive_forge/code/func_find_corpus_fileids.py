import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
def find_corpus_fileids(root, regexp):
    if not isinstance(root, PathPointer):
        raise TypeError('find_corpus_fileids: expected a PathPointer')
    regexp += '$'
    if isinstance(root, ZipFilePathPointer):
        fileids = [name[len(root.entry):] for name in root.zipfile.namelist() if not name.endswith('/')]
        items = [name for name in fileids if re.match(regexp, name)]
        return sorted(items)
    elif isinstance(root, FileSystemPathPointer):
        items = []
        for dirname, subdirs, fileids in os.walk(root.path):
            prefix = ''.join(('%s/' % p for p in _path_from(root.path, dirname)))
            items += [prefix + fileid for fileid in fileids if re.match(regexp, prefix + fileid)]
            if '.svn' in subdirs:
                subdirs.remove('.svn')
        return sorted(items)
    else:
        raise AssertionError("Don't know how to handle %r" % root)