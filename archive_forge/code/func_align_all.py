import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts
def align_all(self, parallel_corpus):
    for sentence_pair in parallel_corpus:
        self.align(sentence_pair)