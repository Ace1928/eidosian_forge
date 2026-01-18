import os
import re
import subprocess
import sys
import tempfile
from nltk.data import ZipFilePathPointer
from nltk.internals import find_dir
from nltk.tokenize.api import TokenizerI
@staticmethod
def parse_repp_outputs(repp_output):
    """
        This module parses the tri-tuple format that REPP outputs using the
        "--format triple" option and returns an generator with tuple of string
        tokens.

        :param repp_output:
        :type repp_output: type
        :return: an iterable of the tokenized sentences as tuples of strings
        :rtype: iter(tuple)
        """
    line_regex = re.compile('^\\((\\d+), (\\d+), (.+)\\)$', re.MULTILINE)
    for section in repp_output.split('\n\n'):
        words_with_positions = [(token, int(start), int(end)) for start, end, token in line_regex.findall(section)]
        words = tuple((t[2] for t in words_with_positions))
        yield words_with_positions