import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
def iter_texts(self):
    for message in self.messages:
        yield (message['role'] == 'user', message['content'])