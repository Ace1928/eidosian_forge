from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.schema import RUN_KEY
def serialize_outputs(self, outputs: Dict) -> str:
    if not outputs.get('generations'):
        raise ValueError('Cannot evaluate LLM Run without generations.')
    generations: List[Dict] = outputs['generations']
    if not generations:
        raise ValueError('Cannot evaluate LLM run with empty generations.')
    first_generation: Dict = generations[0]
    if isinstance(first_generation, list):
        first_generation = first_generation[0]
    if 'message' in first_generation:
        output_ = self.serialize_chat_messages([first_generation['message']])
    else:
        output_ = first_generation['text']
    return output_