import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
def names_demo_features(name):
    features = {}
    features['alwayson'] = True
    features['startswith'] = name[0].lower()
    features['endswith'] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count(%s)' % letter] = name.lower().count(letter)
        features['has(%s)' % letter] = letter in name.lower()
    return features