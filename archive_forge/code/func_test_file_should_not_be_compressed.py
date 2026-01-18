import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
def test_file_should_not_be_compressed(self):
    """
        Is corpus_file a compressed file?
        """
    with tempfile.NamedTemporaryFile(suffix='.bz2') as fp:
        self.assertRaises(TypeError, word2vec.Word2Vec, (None, fp.name))