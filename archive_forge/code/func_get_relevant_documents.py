from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
def get_relevant_documents(self, query: str, *, callbacks: Callbacks=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, run_name: Optional[str]=None, **kwargs: Any) -> List[Document]:
    """Retrieve documents relevant to a query.

        Users should favor using `.invoke` or `.batch` rather than
        `get_relevant_documents directly`.

        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            run_name: Optional name for the run.

        Returns:
            List of relevant documents
        """
    from langchain_core.callbacks.manager import CallbackManager
    callback_manager = CallbackManager.configure(callbacks, None, verbose=kwargs.get('verbose', False), inheritable_tags=tags, local_tags=self.tags, inheritable_metadata=metadata, local_metadata=self.metadata)
    run_manager = callback_manager.on_retriever_start(dumpd(self), query, name=run_name, run_id=kwargs.pop('run_id', None))
    try:
        _kwargs = kwargs if self._expects_other_args else {}
        if self._new_arg_supported:
            result = self._get_relevant_documents(query, run_manager=run_manager, **_kwargs)
        else:
            result = self._get_relevant_documents(query, **_kwargs)
    except Exception as e:
        run_manager.on_retriever_error(e)
        raise e
    else:
        run_manager.on_retriever_end(result)
        return result