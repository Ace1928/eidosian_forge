import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def sentid(self, utterance):
    return utterance.split('/')[1]