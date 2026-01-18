from mlflow.metrics import genai
from mlflow.metrics.base import (
from mlflow.metrics.metric_definitions import (
from mlflow.models import (
from mlflow.utils.annotations import experimental
@experimental
def flesch_kincaid_grade_level() -> EvaluationMetric:
    """
    This function will create a metric for calculating `flesch kincaid grade level`_ using
    `textstat`_.

    This metric outputs a number that approximates the grade level needed to comprehend the text,
    which will likely range from around 0 to 15 (although it is not limited to this range).

    Aggregations calculated for this metric:
        - mean

    .. _flesch kincaid grade level:
        https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level
    .. _textstat: https://pypi.org/project/textstat/
    """
    return make_metric(eval_fn=_flesch_kincaid_eval_fn, greater_is_better=False, name='flesch_kincaid_grade_level', version='v1')