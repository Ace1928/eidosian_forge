import os
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add, and_
from nltk.data import show_cfg
from nltk.inference.mace import MaceCommand
from nltk.inference.prover9 import Prover9Command
from nltk.parse import load_parser
from nltk.parse.malt import MaltParser
from nltk.sem.drt import AnaphoraResolutionException, resolve_anaphora
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Expression
from nltk.tag import RegexpTagger
def load_fol(s):
    """
    Temporarily duplicated from ``nltk.sem.util``.
    Convert a  file of first order formulas into a list of ``Expression`` objects.

    :param s: the contents of the file
    :type s: str
    :return: a list of parsed formulas.
    :rtype: list(Expression)
    """
    statements = []
    for linenum, line in enumerate(s.splitlines()):
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        try:
            statements.append(Expression.fromstring(line))
        except Exception as e:
            raise ValueError(f'Unable to parse line {linenum}: {line}') from e
    return statements