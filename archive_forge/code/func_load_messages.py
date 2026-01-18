from __future__ import annotations
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Type
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def load_messages(self) -> None:
    """Retrieve the messages from Cosmos"""
    if not self._container:
        raise ValueError('Container not initialized')
    try:
        from azure.cosmos.exceptions import CosmosHttpResponseError
    except ImportError as exc:
        raise ImportError('You must install the azure-cosmos package to use the CosmosDBChatMessageHistory.Please install it with `pip install azure-cosmos`.') from exc
    try:
        item = self._container.read_item(item=self.session_id, partition_key=self.user_id)
    except CosmosHttpResponseError:
        logger.info('no session found')
        return
    if 'messages' in item and len(item['messages']) > 0:
        self.messages = messages_from_dict(item['messages'])