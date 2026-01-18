import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
def squad(preds: PREDS_TYPE, target: TARGETS_TYPE) -> Dict[str, Tensor]:
    """Calculate `SQuAD Metric`_ .

    Args:
        preds: A Dictionary or List of Dictionary-s that map `id` and `prediction_text` to the respective values.

            Example prediction:

            .. code-block:: python

                {"prediction_text": "TorchMetrics is awesome", "id": "123"}

        target: A Dictionary or List of Dictionary-s that contain the `answers` and `id` in the SQuAD Format.

            Example target:

            .. code-block:: python

                {
                    'answers': [{'answer_start': [1], 'text': ['This is a test answer']}],
                    'id': '1',
                }

            Reference SQuAD Format:

            .. code-block:: python

                {
                    'answers': {'answer_start': [1], 'text': ['This is a test text']},
                    'context': 'This is a test context.',
                    'id': '1',
                    'question': 'Is this a test?',
                    'title': 'train test'
                }


    Return:
        Dictionary containing the F1 score, Exact match score for the batch.

    Example:
        >>> from torchmetrics.functional.text.squad import squad
        >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]},"id": "56e10a3be3433e1400422b22"}]
        >>> squad(preds, target)
        {'exact_match': tensor(100.), 'f1': tensor(100.)}

    Raises:
        KeyError:
            If the required keys are missing in either predictions or targets.

    References:
        [1] SQuAD: 100,000+ Questions for Machine Comprehension of Text by Pranav Rajpurkar, Jian Zhang, Konstantin
        Lopyrev, Percy Liang `SQuAD Metric`_ .

    """
    preds_dict, target_dict = _squad_input_check(preds, target)
    f1, exact_match, total = _squad_update(preds_dict, target_dict)
    return _squad_compute(f1, exact_match, total)