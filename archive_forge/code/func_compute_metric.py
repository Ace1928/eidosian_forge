from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
def compute_metric(self, a: str, b: str) -> float:
    """
        Compute the distance between two strings.

        Args:
            a (str): The first string.
            b (str): The second string.

        Returns:
            float: The distance between the two strings.
        """
    return self.metric(a, b)