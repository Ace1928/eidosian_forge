import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
def mark_processed(self):
    """
        This is a legacy method, as the Conversation no longer distinguishes between processed and unprocessed user
        input. We set a counter here to keep behaviour mostly backward-compatible, but in general you should just read
        the messages directly when writing new code.
        """
    self._num_processed_user_inputs = len(self._user_messages)