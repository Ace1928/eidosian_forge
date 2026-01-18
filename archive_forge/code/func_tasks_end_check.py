import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def tasks_end_check(self, histories, model_turn=True):
    """
        Check if the current generation sequences have finished.
        """
    for history in histories:
        if not history.completed:
            truncated, ended = self.task_end_check(history, model_turn=model_turn)
            if ended:
                history.complete(truncated=truncated)
    return histories