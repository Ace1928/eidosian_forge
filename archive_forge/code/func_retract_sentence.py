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
def retract_sentence(self, sentence, verbose=True):
    """
        Remove a sentence from the current discourse.

        Updates ``self._input``, ``self._sentences`` and ``self._readings``.
        :param sentence: An input sentence
        :type sentence: str
        :param verbose: If ``True``,  report on the updated list of sentences.
        """
    try:
        self._input.remove(sentence)
    except ValueError:
        print("Retraction failed. The sentence '%s' is not part of the current discourse:" % sentence)
        self.sentences()
        return None
    self._sentences = {'s%s' % i: sent for i, sent in enumerate(self._input)}
    self.readings(verbose=False)
    if verbose:
        print('Current sentences are ')
        self.sentences()