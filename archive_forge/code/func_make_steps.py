from collections import OrderedDict
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
def make_steps():
    return [('sel', SelectFromModel(Perceptron(random_state=None))), ('clf', Perceptron(random_state=None))]