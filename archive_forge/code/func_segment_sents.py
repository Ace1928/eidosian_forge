import json
import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import (
from nltk.tokenize.api import TokenizerI
def segment_sents(self, sentences):
    """ """
    encoding = self._encoding
    _input_fh, self._input_file_path = tempfile.mkstemp(text=True)
    _input_fh = os.fdopen(_input_fh, 'wb')
    _input = '\n'.join((' '.join(x) for x in sentences))
    if isinstance(_input, str) and encoding:
        _input = _input.encode(encoding)
    _input_fh.write(_input)
    _input_fh.close()
    cmd = [self._java_class, '-loadClassifier', self._model, '-keepAllWhitespaces', self._keep_whitespaces, '-textFile', self._input_file_path]
    if self._sihan_corpora_dict is not None:
        cmd.extend(['-serDictionary', self._dict, '-sighanCorporaDict', self._sihan_corpora_dict, '-sighanPostProcessing', self._sihan_post_processing])
    stdout = self._execute(cmd)
    os.unlink(self._input_file_path)
    return stdout