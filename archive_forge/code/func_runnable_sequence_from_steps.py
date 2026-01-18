import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def runnable_sequence_from_steps(steps):
    """Construct a RunnableSequence from steps.

    Args:
        steps: List of steps to construct the RunnableSequence from.
    """
    from langchain.schema.runnable import RunnableSequence
    if len(steps) < 2:
        raise ValueError(f'RunnableSequence must have at least 2 steps, got {len(steps)}.')
    first, *middle, last = steps
    return RunnableSequence(first=first, middle=middle, last=last)