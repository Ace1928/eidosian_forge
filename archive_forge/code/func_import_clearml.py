from __future__ import annotations
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def import_clearml() -> Any:
    """Import the clearml python package and raise an error if it is not installed."""
    try:
        import clearml
    except ImportError:
        raise ImportError('To use the clearml callback manager you need to have the `clearml` python package installed. Please install it with `pip install clearml`')
    return clearml