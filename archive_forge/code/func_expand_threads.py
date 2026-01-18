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
def expand_threads(self, thread_id, threads=None):
    """
        Given a thread ID, find the list of ``logic.Expression`` objects corresponding to the reading IDs in that thread.

        :param thread_id: thread ID
        :type thread_id: str
        :param threads: a mapping from thread IDs to lists of reading IDs
        :type threads: dict
        :return: A list of pairs ``(rid, reading)`` where reading is the ``logic.Expression`` associated with a reading ID
        :rtype: list of tuple
        """
    if threads is None:
        threads = self._threads
    return [(rid, self._readings[sid][rid]) for rid in threads[thread_id] for sid in rid.split('-')[:1]]