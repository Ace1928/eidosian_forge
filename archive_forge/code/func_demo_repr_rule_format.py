import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_repr_rule_format():
    """
    Exemplify repr(Rule) (see also str(Rule) and Rule.format("verbose"))
    """
    postag(ruleformat='repr')