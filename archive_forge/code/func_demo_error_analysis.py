import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def demo_error_analysis():
    """
    Writes a file with context for each erroneous word after tagging testing data
    """
    postag(error_output='errors.txt')