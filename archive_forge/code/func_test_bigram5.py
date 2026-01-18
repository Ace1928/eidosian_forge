from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def test_bigram5():
    b = BigramCollocationFinder.from_words(SENT, window_size=5)
    assert sorted(b.ngram_fd.items()) == sorted([(('a', 'test'), 4), (('is', 'a'), 4), (('this', 'is'), 4), (('is', 'test'), 3), (('this', 'a'), 3), (('a', 'a'), 1), (('is', 'is'), 1), (('test', 'test'), 1), (('this', 'this'), 1)])
    assert sorted(b.word_fd.items()) == sorted([('a', 2), ('is', 2), ('test', 2), ('this', 2)])
    n_word_fd = sum(b.word_fd.values())
    n_ngram_fd = (sum(b.ngram_fd.values()) + 4 + 3 + 2 + 1) / 4.0
    assert len(SENT) == n_word_fd == n_ngram_fd
    assert close_enough(sorted(b.score_ngrams(BigramAssocMeasures.pmi)), sorted([(('a', 'test'), 1.0), (('is', 'a'), 1.0), (('this', 'is'), 1.0), (('is', 'test'), 0.5849625007211562), (('this', 'a'), 0.5849625007211562), (('a', 'a'), -1.0), (('is', 'is'), -1.0), (('test', 'test'), -1.0), (('this', 'this'), -1.0)]))