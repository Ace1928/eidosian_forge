import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
def stream_speech(self, query: str) -> None:
    """Stream the text as speech as it is generated.
        Play the text in your speakers."""
    elevenlabs = _import_elevenlabs()
    speech_stream = elevenlabs.generate(text=query, model=self.model, stream=True)
    elevenlabs.stream(speech_stream)