import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def set_bin_dir(self, bin_dir, verbose=False):
    self._candc_bin = self._find_binary('candc', bin_dir, verbose)
    self._candc_models_path = os.path.normpath(os.path.join(self._candc_bin[:-5], '../models'))
    self._boxer_bin = self._find_binary('boxer', bin_dir, verbose)