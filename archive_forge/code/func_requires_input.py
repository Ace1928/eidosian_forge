from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
@property
def requires_input(self) -> bool:
    """
        This evaluator does not require input.
        """
    return False