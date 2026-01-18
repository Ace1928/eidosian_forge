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
def test_load_old_models_2_x(self):
    """Test loading 2.x models"""
    old_versions = ['2.0.0', '2.1.0', '2.2.0', '2.3.0']
    for old_version in old_versions:
        self._check_old_version(old_version)