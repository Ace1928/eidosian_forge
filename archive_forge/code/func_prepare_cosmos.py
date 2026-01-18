from __future__ import annotations
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Type
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def prepare_cosmos(self) -> None:
    """Prepare the CosmosDB client.

        Use this function or the context manager to make sure your database is ready.
        """
    try:
        from azure.cosmos import PartitionKey
    except ImportError as exc:
        raise ImportError('You must install the azure-cosmos package to use the CosmosDBChatMessageHistory.Please install it with `pip install azure-cosmos`.') from exc
    database = self._client.create_database_if_not_exists(self.cosmos_database)
    self._container = database.create_container_if_not_exists(self.cosmos_container, partition_key=PartitionKey('/user_id'), default_ttl=self.ttl)
    self.load_messages()