import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
def add_message(self, message: Dict[str, str]):
    if not set(message.keys()) == {'role', 'content'}:
        raise ValueError("Message should contain only 'role' and 'content' keys!")
    if message['role'] not in ('user', 'assistant', 'system'):
        raise ValueError("Only 'user', 'assistant' and 'system' roles are supported for now!")
    self.messages.append(message)