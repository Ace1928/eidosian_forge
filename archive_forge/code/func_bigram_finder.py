import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
def bigram_finder(self):
    """Constructs a bigram collocation finder with the bigram and unigram
        data from this finder. Note that this does not include any filtering
        applied to this finder.
        """
    return BigramCollocationFinder(self.word_fd, self.bigram_fd)