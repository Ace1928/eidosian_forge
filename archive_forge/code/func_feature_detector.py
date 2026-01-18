import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
def feature_detector(self, tokens, index, history):
    word = tokens[index]
    if index == 0:
        prevword = prevprevword = None
        prevtag = prevprevtag = None
    elif index == 1:
        prevword = tokens[index - 1].lower()
        prevprevword = None
        prevtag = history[index - 1]
        prevprevtag = None
    else:
        prevword = tokens[index - 1].lower()
        prevprevword = tokens[index - 2].lower()
        prevtag = history[index - 1]
        prevprevtag = history[index - 2]
    if re.match('[0-9]+(\\.[0-9]*)?|[0-9]*\\.[0-9]+$', word):
        shape = 'number'
    elif re.match('\\W+$', word):
        shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        shape = 'upcase'
    elif re.match('[a-z]+$', word):
        shape = 'downcase'
    elif re.match('\\w+$', word):
        shape = 'mixedcase'
    else:
        shape = 'other'
    features = {'prevtag': prevtag, 'prevprevtag': prevprevtag, 'word': word, 'word.lower': word.lower(), 'suffix3': word.lower()[-3:], 'suffix2': word.lower()[-2:], 'suffix1': word.lower()[-1:], 'prevprevword': prevprevword, 'prevword': prevword, 'prevtag+word': f'{prevtag}+{word.lower()}', 'prevprevtag+word': f'{prevprevtag}+{word.lower()}', 'prevword+word': f'{prevword}+{word.lower()}', 'shape': shape}
    return features