from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union
from warnings import warn
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.config import run_in_executor
from langchain.chains.base import Chain
def evaluate_string_pairs(self, *, prediction: str, prediction_b: str, reference: Optional[str]=None, input: Optional[str]=None, **kwargs: Any) -> dict:
    """Evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (Optional[str], optional): The expected output / reference string.
            input (Optional[str], optional): The input string.
            **kwargs: Additional keyword arguments, such as callbacks and optional reference strings.
        Returns:
            dict: A dictionary containing the preference, scores, and/or other information.
        """
    self._check_evaluation_args(reference=reference, input=input)
    return self._evaluate_string_pairs(prediction=prediction, prediction_b=prediction_b, reference=reference, input=input, **kwargs)