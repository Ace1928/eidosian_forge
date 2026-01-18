import math
from . import _catboost
from .core import CatBoost, CatBoostError
from .utils import _import_matplotlib
def explain_features(model):
    _check_model(model)
    return _catboost.explain_features(model._object)