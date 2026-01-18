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
def change_var(self, var):
    return BoxerNamed(self.discourse_id, self.sent_index, self.word_indices, var, self.name, self.type, self.sense)