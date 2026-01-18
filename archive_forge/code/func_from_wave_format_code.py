import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
from langchain_core.runnables import RunnableConfig, RunnableSerializable
@classmethod
def from_wave_format_code(cls, format_code: int) -> 'RivaAudioEncoding':
    """Return the audio encoding specified by the format code in the wave file.

        ref: https://mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
        """
    try:
        return {1: cls.LINEAR_PCM, 6: cls.ALAW, 7: cls.MULAW}[format_code]
    except KeyError as err:
        raise NotImplementedError(f'The following wave file format code is not supported by Riva: {format_code}') from err