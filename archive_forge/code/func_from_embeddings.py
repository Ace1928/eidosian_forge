from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def from_embeddings(cls: Type[Kinetica], text_embeddings: List[Tuple[str, List[float]]], embedding: Embeddings, metadatas: Optional[List[dict]]=None, config: KineticaSettings=KineticaSettings(), dimensions: int=Dimension.OPENAI, collection_name: str=_LANGCHAIN_DEFAULT_COLLECTION_NAME, distance_strategy: DistanceStrategy=DEFAULT_DISTANCE_STRATEGY, ids: Optional[List[str]]=None, pre_delete_collection: bool=False, **kwargs: Any) -> Kinetica:
    """Adds the embeddings passed in to the vector store and returns it

        Args:
            cls (Type[Kinetica]): Kinetica class
            text_embeddings (List[Tuple[str, List[float]]]): A list of texts
                            and the embeddings
            embedding (Embeddings): List of embeddings
            metadatas (Optional[List[dict]], optional): List of dicts, JSON describing
                        the texts/documents. Defaults to None.
            config (KineticaSettings): a `KineticaSettings` instance
            dimensions (int, optional): Dimension for the vector data, if not passed a
                        default will be used. Defaults to Dimension.OPENAI.
            collection_name (str, optional): Kinetica schema name.
                        Defaults to _LANGCHAIN_DEFAULT_COLLECTION_NAME.
            distance_strategy (DistanceStrategy, optional): Distance strategy
                        e.g., l2, cosine etc.. Defaults to DEFAULT_DISTANCE_STRATEGY.
            ids (Optional[List[str]], optional): A list of UUIDs for each text/document.
                        Defaults to None.
            pre_delete_collection (bool, optional): Indicates whether the
                        Kinetica schema is to be deleted or not. Defaults to False.

        Returns:
            Kinetica: a `Kinetica` instance
        """
    texts = [t[0] for t in text_embeddings]
    embeddings = [t[1] for t in text_embeddings]
    dimensions = len(embeddings[0])
    return cls.__from(texts=texts, embeddings=embeddings, embedding=embedding, dimensions=dimensions, config=config, metadatas=metadatas, ids=ids, collection_name=collection_name, distance_strategy=distance_strategy, pre_delete_collection=pre_delete_collection, **kwargs)