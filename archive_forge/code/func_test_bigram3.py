from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def test_bigram3():
    b = BigramCollocationFinder.from_words(SENT, window_size=3)
    assert sorted(b.ngram_fd.items()) == sorted([(('a', 'test'), 3), (('is', 'a'), 3), (('this', 'is'), 3), (('a', 'a'), 1), (('is', 'is'), 1), (('test', 'test'), 1), (('this', 'this'), 1)])
    assert sorted(b.word_fd.items()) == sorted([('a', 2), ('is', 2), ('test', 2), ('this', 2)])
    assert len(SENT) == sum(b.word_fd.values()) == (sum(b.ngram_fd.values()) + 2 + 1) / 2.0
    assert close_enough(sorted(b.score_ngrams(BigramAssocMeasures.pmi)), sorted([(('a', 'test'), 1.584962500721156), (('is', 'a'), 1.584962500721156), (('this', 'is'), 1.584962500721156), (('a', 'a'), 0.0), (('is', 'is'), 0.0), (('test', 'test'), 0.0), (('this', 'this'), 0.0)]))