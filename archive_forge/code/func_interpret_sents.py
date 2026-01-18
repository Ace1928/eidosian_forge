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
def interpret_sents(self, inputs, discourse_ids=None, question=False, verbose=False):
    """
        Use Boxer to give a first order representation.

        :param inputs: list of str Input sentences to parse as individual discourses
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_ids: list of str Identifiers to be inserted to each occurrence-indexed predicate.
        :return: list of ``drt.DrtExpression``
        """
    return self.interpret_multi_sents([[input] for input in inputs], discourse_ids, question, verbose)