from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.cohere import BaseCohere
def get_cohere_chat_request(messages: List[BaseMessage], *, connectors: Optional[List[Dict[str, str]]]=None, **kwargs: Any) -> Dict[str, Any]:
    """Get the request for the Cohere chat API.

    Args:
        messages: The messages.
        connectors: The connectors.
        **kwargs: The keyword arguments.

    Returns:
        The request for the Cohere chat API.
    """
    documents = None if 'source_documents' not in kwargs else [{'snippet': doc.page_content, 'id': doc.metadata.get('id') or f'doc-{str(i)}'} for i, doc in enumerate(kwargs['source_documents'])]
    kwargs.pop('source_documents', None)
    maybe_connectors = connectors if documents is None else None
    prompt_truncation = 'AUTO' if documents is not None or connectors is not None else None
    req = {'message': messages[-1].content, 'chat_history': [{'role': get_role(x), 'message': x.content} for x in messages[:-1]], 'documents': documents, 'connectors': maybe_connectors, 'prompt_truncation': prompt_truncation, **kwargs}
    return {k: v for k, v in req.items() if v is not None}