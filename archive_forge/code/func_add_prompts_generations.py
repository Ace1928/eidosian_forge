import os
import warnings
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.outputs import Generation, LLMResult
def add_prompts_generations(self, run_id: str, generations: List[List[Generation]]) -> None:
    tasks = []
    prompts = self.payload[run_id]['prompts']
    model_version = self.payload[run_id]['kwargs'].get('invocation_params', {}).get('model_name')
    for prompt, generation in zip(prompts, generations):
        tasks.append({'data': {self.value: prompt, 'run_id': run_id}, 'predictions': [{'result': [{'from_name': self.from_name, 'to_name': self.to_name, 'type': 'textarea', 'value': {'text': [g.text for g in generation]}}], 'model_version': model_version}]})
    self.ls_project.import_tasks(tasks)