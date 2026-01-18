from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def demo_pos_bw(test=10, supervised=20, unsupervised=10, verbose=True, max_iterations=5):
    print()
    print('Baum-Welch demo for POS tagging')
    print()
    print('Training HMM (supervised, %d sentences)...' % supervised)
    sentences, tag_set, symbols = load_pos(test + supervised + unsupervised)
    symbols = set()
    for sentence in sentences:
        for token in sentence:
            symbols.add(token[_TEXT])
    trainer = HiddenMarkovModelTrainer(tag_set, list(symbols))
    hmm = trainer.train_supervised(sentences[test:test + supervised], estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
    hmm.test(sentences[:test], verbose=verbose)
    print('Training (unsupervised, %d sentences)...' % unsupervised)
    unlabeled = _untag(sentences[test + supervised:])
    hmm = trainer.train_unsupervised(unlabeled, model=hmm, max_iterations=max_iterations)
    hmm.test(sentences[:test], verbose=verbose)