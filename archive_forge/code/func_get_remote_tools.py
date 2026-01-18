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
def get_remote_tools(organization='huggingface-tools'):
    if is_offline_mode():
        logger.info('You are in offline mode, so remote tools are not available.')
        return {}
    spaces = list_spaces(author=organization)
    tools = {}
    for space_info in spaces:
        repo_id = space_info.id
        resolved_config_file = hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type='space')
        with open(resolved_config_file, encoding='utf-8') as reader:
            config = json.load(reader)
        task = repo_id.split('/')[-1]
        tools[config['name']] = PreTool(task=task, description=config['description'], repo_id=repo_id)
    return tools