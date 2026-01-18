from __future__ import annotations
import json
import re
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from sqlalchemy.pool import QueuePool
from langchain_community.vectorstores.utils import DistanceStrategy
def add_images(self, uris: List[str], metadatas: Optional[List[dict]]=None, embeddings: Optional[List[List[float]]]=None, **kwargs: Any) -> List[str]:
    """Run images through the embeddings and add to the vectorstore.

        Args:
            uris List[str]: File path to images.
                Each URI will be added to the vectorstore as document content.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            List[str]: empty list
        """
    if embeddings is None and self.embedding is not None and hasattr(self.embedding, 'embed_image'):
        embeddings = self.embedding.embed_image(uris=uris)
    return self.add_texts(uris, metadatas, embeddings, **kwargs)