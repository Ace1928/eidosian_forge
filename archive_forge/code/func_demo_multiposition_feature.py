import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_multiposition_feature():
    """
    The feature/s of a template takes a list of positions
    relative to the current word where the feature should be
    looked for, conceptually joined by logical OR. For instance,
    Pos([-1, 1]), given a value V, will hold whenever V is found
    one step to the left and/or one step to the right.

    For contiguous ranges, a 2-arg form giving inclusive end
    points can also be used: Pos(-3, -1) is the same as the arg
    below.
    """
    postag(templates=[Template(Pos([-3, -2, -1]))])