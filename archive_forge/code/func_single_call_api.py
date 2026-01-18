from __future__ import annotations
import json
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Union
import langchain.chains
import pydantic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AIMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
def single_call_api(self, callback_handlers: Optional[List[BaseCallbackHandler]]):
    from langchain.schema import BaseRetriever
    from mlflow.langchain.utils import lc_runnables_types
    if isinstance(self.lc_model, BaseRetriever):
        docs = self.lc_model.get_relevant_documents(**self.request_json)
        response = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in docs]
    elif isinstance(self.lc_model, lc_runnables_types()):

        def _predict_single_input(single_input):
            if self.stream:
                return self.lc_model.stream(single_input, config={'callbacks': callback_handlers})
            return self.lc_model.invoke(single_input, config={'callbacks': callback_handlers})
        if isinstance(self.request_json, dict):
            try:
                response = _predict_single_input(self.request_json)
            except TypeError as e:
                _logger.warning(f'Failed to invoke {self.lc_model.__class__.__name__} with {self.request_json}. Error: {e!r}. Trying to invoke with the first value of the dictionary.')
                self.request_json = next(iter(self.request_json.values()))
                prepared_request_json, did_perform_chat_conversion = APIRequest._transform_request_json_for_chat_if_necessary(self.request_json, self.lc_model)
                self.did_perform_chat_conversion = did_perform_chat_conversion
                response = _predict_single_input(prepared_request_json)
        else:
            response = _predict_single_input(self.request_json)
        if self.did_perform_chat_conversion or self.convert_chat_responses:
            if self.stream:
                response = APIRequest._try_transform_response_iter_to_chat_format(response)
            else:
                response = APIRequest._try_transform_response_to_chat_format(response)
    else:
        response = self.lc_model(self.request_json, return_only_outputs=True, callbacks=callback_handlers)
        if self.did_perform_chat_conversion or self.convert_chat_responses:
            response = APIRequest._try_transform_response_to_chat_format(response)
        elif len(response) == 1:
            response = response.popitem()[1]
        else:
            self._prepare_to_serialize(response)
    return response