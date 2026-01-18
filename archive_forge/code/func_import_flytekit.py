from __future__ import annotations
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def import_flytekit() -> Tuple[flytekit, renderer]:
    """Import flytekit and flytekitplugins-deck-standard."""
    try:
        import flytekit
        from flytekitplugins.deck import renderer
    except ImportError:
        raise ImportError('To use the flyte callback manager you needto have the `flytekit` and `flytekitplugins-deck-standard`packages installed. Please install them with `pip install flytekit`and `pip install flytekitplugins-deck-standard`.')
    return (flytekit, renderer)