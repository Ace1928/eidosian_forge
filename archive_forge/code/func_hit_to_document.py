from __future__ import annotations
import logging
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def hit_to_document(hit: resources_pb2.Hit) -> Tuple[Document, float]:
    metadata = json_format.MessageToDict(hit.input.data.metadata)
    h = dict(self._auth.metadata)
    request = requests.get(hit.input.data.text.url, headers=h)
    request.encoding = request.apparent_encoding
    requested_text = request.text
    logger.debug(f'\tScore {hit.score:.2f} for annotation: {hit.annotation.id}                off input: {hit.input.id}, text: {requested_text[:125]}')
    return (Document(page_content=requested_text, metadata=metadata), hit.score)