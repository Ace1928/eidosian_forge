from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
import openai
from typing_extensions import override
def query_with_system_prompt_with_retries(self, system_prompt: str, prompt: str) -> str:
    return self._query_with_retries(self.query_with_system_prompt, system_prompt, prompt)