from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
@property
def evaluation_name(self) -> str:
    """
        Get the evaluation name.

        Returns:
            str: The evaluation name.
        """
    return f'pairwise_{self.distance.value}_distance'