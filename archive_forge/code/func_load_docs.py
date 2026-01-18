import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def load_docs(self, query: str) -> List[Document]:
    return list(self.lazy_load_docs(query=query))