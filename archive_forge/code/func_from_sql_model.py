import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sqlalchemy import Column, Integer, Text, create_engine
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
from sqlalchemy.orm import sessionmaker
def from_sql_model(self, sql_message: Any) -> BaseMessage:
    return messages_from_dict([json.loads(sql_message.message)])[0]