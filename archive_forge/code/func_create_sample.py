import inspect
import types
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ..data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
@staticmethod
def create_sample(question: Union[str, List[str]], context: Union[str, List[str]]) -> Union[SquadExample, List[SquadExample]]:
    """
        QuestionAnsweringPipeline leverages the [`SquadExample`] internally. This helper method encapsulate all the
        logic for converting question(s) and context(s) to [`SquadExample`].

        We currently support extractive question answering.

        Arguments:
            question (`str` or `List[str]`): The question(s) asked.
            context (`str` or `List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of [`SquadExample`]: The corresponding [`SquadExample`] grouping question and context.
        """
    if isinstance(question, list):
        return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
    else:
        return SquadExample(None, question, context, None, None, None)