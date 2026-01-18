from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
@classmethod
def from_files(cls: Type[Vectara], files: List[str], embedding: Optional[Embeddings]=None, metadatas: Optional[List[dict]]=None, **kwargs: Any) -> Vectara:
    """Construct Vectara wrapper from raw documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Vectara
                vectara = Vectara.from_files(
                    files_list,
                    vectara_customer_id=customer_id,
                    vectara_corpus_id=corpus_id,
                    vectara_api_key=api_key,
                )
        """
    vectara = cls(**kwargs)
    vectara.add_files(files, metadatas)
    return vectara