import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict
import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate
def generate_one(self, prompt, stop):
    encoded_inputs = self.tokenizer(prompt, return_tensors='pt').to(self._model_device)
    src_len = encoded_inputs['input_ids'].shape[1]
    stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop, self.tokenizer)])
    outputs = self.model.generate(encoded_inputs['input_ids'], max_new_tokens=200, stopping_criteria=stopping_criteria)
    result = self.tokenizer.decode(outputs[0].tolist()[src_len:])
    for stop_seq in stop:
        if result.endswith(stop_seq):
            result = result[:-len(stop_seq)]
    return result