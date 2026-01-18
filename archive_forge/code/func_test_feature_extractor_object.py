import pytest
from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus
def test_feature_extractor_object(self):
    rtepair = rte_corpus.pairs(['rte3_dev.xml'])[33]
    extractor = RTEFeatureExtractor(rtepair)
    assert extractor.hyp_words == {'member', 'China', 'SCO.'}
    assert extractor.overlap('word') == set()
    assert extractor.overlap('ne') == {'China'}
    assert extractor.hyp_extra('word') == {'member'}