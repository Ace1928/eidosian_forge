import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_template_statistics():
    """
    Show aggregate statistics per template. Little used templates are
    candidates for deletion, much used templates may possibly be refined.

    Deleting unused templates is mostly about saving time and/or space:
    training is basically O(T) in the number of templates T
    (also in terms of memory usage, which often will be the limiting factor).
    """
    postag(incremental_stats=True, template_stats=True)