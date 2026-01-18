from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
import openai
from typing_extensions import override
def query_with_retries(self, prompt: str) -> str:
    return self._query_with_retries(self.query, prompt)