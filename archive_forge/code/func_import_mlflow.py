import logging
import os
import random
import string
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.utils import get_from_dict_or_env
from langchain_community.callbacks.utils import (
def import_mlflow() -> Any:
    """Import the mlflow python package and raise an error if it is not installed."""
    try:
        import mlflow
    except ImportError:
        raise ImportError('To use the mlflow callback manager you need to have the `mlflow` python package installed. Please install it with `pip install mlflow>=2.3.0`')
    return mlflow