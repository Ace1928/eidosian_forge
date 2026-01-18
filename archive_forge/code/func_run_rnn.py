from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
def run_rnn(self, _tokens: List[str], newline_adj: int=0) -> Tuple[List[float], int]:
    tokens = [int(x) for x in _tokens]
    token_len = len(tokens)
    self.model_tokens += tokens
    out, self.model_state = self.model.forward(tokens, self.model_state)
    return (out, token_len)