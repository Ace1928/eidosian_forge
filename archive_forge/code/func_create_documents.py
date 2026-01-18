from __future__ import annotations
import copy
import json
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
def create_documents(self, texts: List[Dict], convert_lists: bool=False, metadatas: Optional[List[dict]]=None) -> List[Document]:
    """Create documents from a list of json objects (Dict)."""
    _metadatas = metadatas or [{}] * len(texts)
    documents = []
    for i, text in enumerate(texts):
        for chunk in self.split_text(json_data=text, convert_lists=convert_lists):
            metadata = copy.deepcopy(_metadatas[i])
            new_doc = Document(page_content=chunk, metadata=metadata)
            documents.append(new_doc)
    return documents