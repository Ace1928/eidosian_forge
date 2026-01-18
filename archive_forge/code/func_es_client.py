import logging
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@staticmethod
def es_client(*, es_url: Optional[str]=None, username: Optional[str]=None, password: Optional[str]=None, **kwargs: Optional[dict]) -> 'Elasticsearch':
    try:
        import elasticsearch
    except ImportError:
        raise ImportError('Could not import elasticsearch python package. Please install it with `pip install elasticsearch`.')
    connection_params: Dict[str, Any] = {'hosts': [es_url]}
    if username and password:
        connection_params['http_auth'] = (username, password)
    connection_params.update(kwargs)
    es_client = elasticsearch.Elasticsearch(**connection_params)
    try:
        es_client.info()
    except Exception as e:
        logger.error(f'Error connecting to Elasticsearch: {e}')
        raise e
    return es_client