from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import httpx
import huggingface_hub
import websockets
from packaging import version
from gradio_client import serializing, utils
from gradio_client.exceptions import SerializationSetupError
from gradio_client.utils import (
def insert_state(self, *data) -> tuple:
    data = list(data)
    for i, input_component_type in enumerate(self.input_component_types):
        if input_component_type == utils.STATE_COMPONENT:
            data.insert(i, None)
    return tuple(data)