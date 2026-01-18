from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def get_tool_label(self, tool: ToolRecord, is_complete: bool) -> str:
    """Return the label for an LLMThought that has an associated
        tool.

        Parameters
        ----------
        tool
            The tool's ToolRecord

        is_complete
            True if the thought is complete; False if the thought
            is still receiving input.

        Returns
        -------
        The markdown label for the thought's container.

        """
    input = tool.input_str
    name = tool.name
    emoji = CHECKMARK_EMOJI if is_complete else THINKING_EMOJI
    if name == '_Exception':
        emoji = EXCEPTION_EMOJI
        name = 'Parsing error'
    idx = min([60, len(input)])
    input = input[0:idx]
    if len(tool.input_str) > idx:
        input = input + '...'
    input = input.replace('\n', ' ')
    label = f'{emoji} **{name}:** {input}'
    return label