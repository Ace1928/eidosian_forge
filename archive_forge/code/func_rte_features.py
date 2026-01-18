from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import RegexpTokenizer
def rte_features(rtepair):
    extractor = RTEFeatureExtractor(rtepair)
    features = {}
    features['alwayson'] = True
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    features['neg_txt'] = len(extractor.negwords & extractor.text_words)
    features['neg_hyp'] = len(extractor.negwords & extractor.hyp_words)
    return features