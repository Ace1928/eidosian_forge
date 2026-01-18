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
def interpret_multi_sents(self, inputs, discourse_ids=None, question=False, verbose=False):
    """
        Use Boxer to give a first order representation.

        :param inputs: list of list of str Input discourses to parse
        :param occur_index: bool Should predicates be occurrence indexed?
        :param discourse_ids: list of str Identifiers to be inserted to each occurrence-indexed predicate.
        :return: ``drt.DrtExpression``
        """
    if discourse_ids is not None:
        assert len(inputs) == len(discourse_ids)
        assert reduce(operator.and_, (id is not None for id in discourse_ids))
        use_disc_id = True
    else:
        discourse_ids = list(map(str, range(len(inputs))))
        use_disc_id = False
    candc_out = self._call_candc(inputs, discourse_ids, question, verbose=verbose)
    boxer_out = self._call_boxer(candc_out, verbose=verbose)
    drs_dict = self._parse_to_drs_dict(boxer_out, use_disc_id)
    return [drs_dict.get(id, None) for id in discourse_ids]