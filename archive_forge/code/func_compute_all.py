import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def compute_all(self):
    """Convenience method to perform all computations."""
    self.compute_distances()
    self.compute_distance_gradients()
    self.compute_gradients()
    self.compute_loss()