from __future__ import annotations
import hashlib
import json
import uuid
from itertools import islice
from typing import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.vectorstores import VectorStore
from langchain.indexes.base import NAMESPACE_UUID, RecordManager
@root_validator(pre=True)
def calculate_hashes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Root validator to calculate content and metadata hash."""
    content = values.get('page_content', '')
    metadata = values.get('metadata', {})
    forbidden_keys = ('hash_', 'content_hash', 'metadata_hash')
    for key in forbidden_keys:
        if key in metadata:
            raise ValueError(f'Metadata cannot contain key {key} as it is reserved for internal use.')
    content_hash = str(_hash_string_to_uuid(content))
    try:
        metadata_hash = str(_hash_nested_dict_to_uuid(metadata))
    except Exception as e:
        raise ValueError(f'Failed to hash metadata: {e}. Please use a dict that can be serialized using json.')
    values['content_hash'] = content_hash
    values['metadata_hash'] = metadata_hash
    values['hash_'] = str(_hash_string_to_uuid(content_hash + metadata_hash))
    _uid = values.get('uid', None)
    if _uid is None:
        values['uid'] = values['hash_']
    return values