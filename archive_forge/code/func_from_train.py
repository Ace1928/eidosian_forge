import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
@staticmethod
def from_train(tokens):
    """
        Constructs an ARFF_Formatter instance with class labels and feature
        types determined from the given data. Handles boolean, numeric and
        string (note: not nominal) types.
        """
    labels = {label for tok, label in tokens}
    features = {}
    for tok, label in tokens:
        for fname, fval in tok.items():
            if issubclass(type(fval), bool):
                ftype = '{True, False}'
            elif issubclass(type(fval), (int, float, bool)):
                ftype = 'NUMERIC'
            elif issubclass(type(fval), str):
                ftype = 'STRING'
            elif fval is None:
                continue
            else:
                raise ValueError('Unsupported value type %r' % ftype)
            if features.get(fname, ftype) != ftype:
                raise ValueError('Inconsistent type for %s' % fname)
            features[fname] = ftype
    features = sorted(features.items())
    return ARFF_Formatter(labels, features)