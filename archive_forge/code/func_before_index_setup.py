import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
def before_index_setup(self, client: 'Elasticsearch', text_field: str, vector_query_field: str) -> None:
    if self.model_id:
        client.ingest.put_pipeline(id=self._get_pipeline_name(), description='Embedding pipeline for langchain vectorstore', processors=[{'inference': {'model_id': self.model_id, 'target_field': vector_query_field, 'field_map': {text_field: 'text_field'}, 'inference_config': {'text_expansion': {'results_field': 'tokens'}}}}])