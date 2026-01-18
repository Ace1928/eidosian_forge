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
def evaluate_strings(self, *, prediction: str, reference: Optional[str]=None, input: Optional[str]=None, **kwargs: Any) -> dict:
    """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            **kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """
    self._check_evaluation_args(reference=reference, input=input)
    return self._evaluate_strings(prediction=prediction, reference=reference, input=input, **kwargs)