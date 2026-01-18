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
def handle_condition(self, tok, indices):
    """
        Handle a DRS condition

        :param indices: list of int
        :return: list of ``DrtExpression``
        """
    if tok == 'not':
        return [self._handle_not()]
    if tok == 'or':
        conds = [self._handle_binary_expression(self._make_or_expression)]
    elif tok == 'imp':
        conds = [self._handle_binary_expression(self._make_imp_expression)]
    elif tok == 'eq':
        conds = [self._handle_eq()]
    elif tok == 'prop':
        conds = [self._handle_prop()]
    elif tok == 'pred':
        conds = [self._handle_pred()]
    elif tok == 'named':
        conds = [self._handle_named()]
    elif tok == 'rel':
        conds = [self._handle_rel()]
    elif tok == 'timex':
        conds = self._handle_timex()
    elif tok == 'card':
        conds = [self._handle_card()]
    elif tok == 'whq':
        conds = [self._handle_whq()]
    elif tok == 'duplex':
        conds = [self._handle_duplex()]
    else:
        conds = []
    return sum(([cond(sent_index, word_indices) for cond in conds] for sent_index, word_indices in self._sent_and_word_indices(indices)), [])