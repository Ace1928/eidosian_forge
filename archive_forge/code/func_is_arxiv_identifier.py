import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def is_arxiv_identifier(self, query: str) -> bool:
    """Check if a query is an arxiv identifier."""
    arxiv_identifier_pattern = '\\d{2}(0[1-9]|1[0-2])\\.\\d{4,5}(v\\d+|)|\\d{7}.*'
    for query_item in query[:self.ARXIV_MAX_QUERY_LENGTH].split():
        match_result = re.match(arxiv_identifier_pattern, query_item)
        if not match_result:
            return False
        assert match_result is not None
        if not match_result.group(0) == query_item:
            return False
    return True