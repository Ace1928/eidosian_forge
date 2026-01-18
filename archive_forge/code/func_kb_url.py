import os
from typing import Any, Dict, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
@property
def kb_url(self) -> str:
    return f'{self._config['BACKEND']}/kb/{self._kb}'