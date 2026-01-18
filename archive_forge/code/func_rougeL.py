from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def rougeL() -> EvaluationMetric:
    """
    This function will create a metric for evaluating `rougeL`_.

    The score ranges from 0 to 1, where a higher score indicates higher similarity.
    `rougeL`_ uses unigram based scoring to calculate similarity.

    Aggregations calculated for this metric:
        - mean

    .. _rougeL: https://huggingface.co/spaces/evaluate-metric/rouge
    """
    return make_metric(eval_fn=_rougeL_eval_fn, greater_is_better=True, name='rougeL', version='v1')