import os
import re
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir
from nltk.tokenize.api import TokenizerI
def find_repptokenizer(self, repp_dirname):
    """
        A module to find REPP tokenizer binary and its *repp.set* config file.
        """
    if os.path.exists(repp_dirname):
        _repp_dir = repp_dirname
    else:
        _repp_dir = find_dir(repp_dirname, env_vars=('REPP_TOKENIZER',))
    assert os.path.exists(_repp_dir + '/src/repp')
    assert os.path.exists(_repp_dir + '/erg/repp.set')
    return _repp_dir