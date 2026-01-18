from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
def mape() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `mape`_.

    This metric computes an aggregate score for the mean absolute percentage error for regression.

    .. _mape: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
    """
    return make_metric(eval_fn=_mape_eval_fn, greater_is_better=False, name='mean_absolute_percentage_error')