import logging
import pickle
import random
from collections import defaultdict
from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI
def average_weights(self):
    """Average weights from all iterations."""
    for feat, weights in self.weights.items():
        new_feat_weights = {}
        for clas, weight in weights.items():
            param = (feat, clas)
            total = self._totals[param]
            total += (self.i - self._tstamps[param]) * weight
            averaged = round(total / self.i, 3)
            if averaged:
                new_feat_weights[clas] = averaged
        self.weights[feat] = new_feat_weights