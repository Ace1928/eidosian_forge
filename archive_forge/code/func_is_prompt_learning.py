import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from .utils import CONFIG_NAME, PeftType, TaskType
@property
def is_prompt_learning(self) -> bool:
    """
        Utility method to check if the configuration is for prompt learning.
        """
    return True