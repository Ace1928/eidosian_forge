from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def ari_grade_level() -> EvaluationMetric:
    """
    This function will create a metric for calculating `automated readability index`_ using
    `textstat`_.

    This metric outputs a number that approximates the grade level needed to comprehend the text,
    which will likely range from around 0 to 15 (although it is not limited to this range).

    Aggregations calculated for this metric:
        - mean

    .. _automated readability index: https://en.wikipedia.org/wiki/Automated_readability_index
    .. _textstat: https://pypi.org/project/textstat/
    """
    return make_metric(eval_fn=_ari_eval_fn, greater_is_better=False, name='ari_grade_level', long_name='automated_readability_index_grade_level', version='v1')