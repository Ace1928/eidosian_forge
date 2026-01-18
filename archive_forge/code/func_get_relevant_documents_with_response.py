from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
def get_relevant_documents_with_response(self, query: str) -> Tuple[List[Document], Any]:
    from google.api_core.exceptions import InvalidArgument
    search_request = self._create_search_request(query)
    try:
        response = self._client.search(search_request)
    except InvalidArgument as exc:
        raise type(exc)(exc.message + ' This might be due to engine_data_type not set correctly.')
    if self.engine_data_type == 0:
        chunk_type = 'extractive_answers' if self.get_extractive_answers else 'extractive_segments'
        documents = self._convert_unstructured_search_response(response.results, chunk_type)
    elif self.engine_data_type == 1:
        documents = self._convert_structured_search_response(response.results)
    elif self.engine_data_type in (2, 3):
        chunk_type = 'extractive_answers' if self.get_extractive_answers else 'snippets'
        documents = self._convert_website_search_response(response.results, chunk_type)
    else:
        raise NotImplementedError('Only data store type 0 (Unstructured), 1 (Structured),2 (Website), or 3 (Blended) are supported currently.' + f' Got {self.engine_data_type}')
    return (documents, response)