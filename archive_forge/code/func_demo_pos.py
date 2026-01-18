from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def demo_pos():
    print()
    print('HMM POS tagging demo')
    print()
    print('Training HMM...')
    labelled_sequences, tag_set, symbols = load_pos(20000)
    trainer = HiddenMarkovModelTrainer(tag_set, symbols)
    hmm = trainer.train_supervised(labelled_sequences[10:], estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
    print('Testing...')
    hmm.test(labelled_sequences[:10], verbose=True)