import logging
import sys
from collections import defaultdict
from typing import (
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
    if 'contextual_compression' in serialized['id']:
        self.schema.eval_types.add('contextual_compression')
        self.schema.query = query
        self.schema.context_conciseness_run_id = run_id
    if 'multi_query' in serialized['id']:
        self.schema.eval_types.add('multi_query')
        self.schema.multi_query_run_id = run_id
        self.schema.query = query
    elif 'multi_query' in self.schema.eval_types:
        self.schema.multi_queries.append(query)