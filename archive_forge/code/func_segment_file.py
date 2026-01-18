import json
import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import (
from nltk.tokenize.api import TokenizerI
def segment_file(self, input_file_path):
    """ """
    cmd = [self._java_class, '-loadClassifier', self._model, '-keepAllWhitespaces', self._keep_whitespaces, '-textFile', input_file_path]
    if self._sihan_corpora_dict is not None:
        cmd.extend(['-serDictionary', self._dict, '-sighanCorporaDict', self._sihan_corpora_dict, '-sighanPostProcessing', self._sihan_post_processing])
    stdout = self._execute(cmd)
    return stdout