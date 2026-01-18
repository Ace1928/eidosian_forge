import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def construct_html_from_prompt_and_generation(prompt: str, generation: str) -> Any:
    """Construct an html element from a prompt and a generation.

    Parameters:
        prompt (str): The prompt.
        generation (str): The generation.

    Returns:
        (wandb.Html): The html element."""
    wandb = import_wandb()
    formatted_prompt = prompt.replace('\n', '<br>')
    formatted_generation = generation.replace('\n', '<br>')
    return wandb.Html(f'\n    <p style="color:black;">{formatted_prompt}:</p>\n    <blockquote>\n      <p style="color:green;">\n        {formatted_generation}\n      </p>\n    </blockquote>\n    ', inject=False)