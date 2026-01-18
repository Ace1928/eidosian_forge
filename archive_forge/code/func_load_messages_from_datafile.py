import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@classmethod
def load_messages_from_datafile(cls, sa_datafile: Path) -> List[BaseMessage]:
    """Load a lanchain prompt from a Kinetica context datafile."""
    datafile_dict = _KineticaLlmFileContextParser.parse_dialogue_file(sa_datafile)
    messages = cls._convert_dict_to_messages(datafile_dict)
    return messages