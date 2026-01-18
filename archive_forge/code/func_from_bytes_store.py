from __future__ import annotations
import hashlib
import json
import uuid
from functools import partial
from typing import Callable, List, Optional, Sequence, Union, cast
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.utils.iter import batch_iterate
from langchain.storage.encoder_backed import EncoderBackedStore
@classmethod
def from_bytes_store(cls, underlying_embeddings: Embeddings, document_embedding_cache: ByteStore, *, namespace: str='', batch_size: Optional[int]=None) -> CacheBackedEmbeddings:
    """On-ramp that adds the necessary serialization and encoding to the store.

        Args:
            underlying_embeddings: The embedder to use for embedding.
            document_embedding_cache: The cache to use for storing document embeddings.
            *,
            namespace: The namespace to use for document cache.
                       This namespace is used to avoid collisions with other caches.
                       For example, set it to the name of the embedding model used.
            batch_size: The number of documents to embed between store updates.
        """
    namespace = namespace
    key_encoder = _create_key_encoder(namespace)
    encoder_backed_store = EncoderBackedStore[str, List[float]](document_embedding_cache, key_encoder, _value_serializer, _value_deserializer)
    return cls(underlying_embeddings, encoder_backed_store, batch_size=batch_size)