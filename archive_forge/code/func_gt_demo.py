import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def gt_demo():
    from nltk import corpus
    emma_words = corpus.gutenberg.words('austen-emma.txt')
    fd = FreqDist(emma_words)
    sgt = SimpleGoodTuringProbDist(fd)
    print('{:>18} {:>8}  {:>14}'.format('word', 'frequency', 'SimpleGoodTuring'))
    fd_keys_sorted = (key for key, value in sorted(fd.items(), key=lambda item: item[1], reverse=True))
    for key in fd_keys_sorted:
        print('%18s %8d  %14e' % (key, fd[key], sgt.prob(key)))