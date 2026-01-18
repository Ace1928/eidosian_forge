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
def test_load_old_models_1_x(self):
    """Test loading 1.x models"""
    old_versions = ['1.0.0', '1.0.1']
    for old_version in old_versions:
        self._check_old_version(old_version)