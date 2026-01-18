import math
from . import _catboost
from .core import CatBoost, CatBoostError
from .utils import _import_matplotlib
def calc_features_strength(model):
    explanations = explain_features(model)
    features_strength = [expl.calc_strength() for expl in explanations]
    return features_strength