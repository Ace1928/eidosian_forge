import warnings
from inspect import signature
from math import log
from numbers import Integral, Real
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import Bunch
from ._loss import HalfBinomialLoss
from .base import (
from .isotonic import IsotonicRegression
from .model_selection import check_cv, cross_val_predict
from .preprocessing import LabelEncoder, label_binarize
from .svm import LinearSVC
from .utils import (
from .utils._param_validation import (
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
from .utils._response import _get_response_values, _process_predict_proba
from .utils.metadata_routing import (
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import (
def loss_grad(AB):
    raw_prediction = -(AB[0] * F + AB[1]).astype(dtype=predictions.dtype)
    l, g = bin_loss.loss_gradient(y_true=T, raw_prediction=raw_prediction, sample_weight=sample_weight)
    loss = l.sum()
    grad = np.asarray([-g @ F, -g.sum()], dtype=np.float64)
    return (loss, grad)