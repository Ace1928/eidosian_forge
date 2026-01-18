from __future__ import annotations
import re
from abc import abstractmethod
from collections import deque
from typing import AsyncIterator, Deque, Iterator, List, TypeVar, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
def parse_iter(self, text: str) -> Iterator[re.Match]:
    """Parse the output of an LLM call."""
    return re.finditer(self.pattern, text, re.MULTILINE)