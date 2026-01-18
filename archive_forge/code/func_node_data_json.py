from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def node_data_json(node: Node, *, with_schemas: bool=False) -> Dict[str, Union[str, Dict[str, Any]]]:
    from langchain_core.load.serializable import to_json_not_implemented
    from langchain_core.runnables.base import Runnable, RunnableSerializable
    if isinstance(node.data, RunnableSerializable):
        return {'type': 'runnable', 'data': {'id': node.data.lc_id(), 'name': node.data.get_name()}}
    elif isinstance(node.data, Runnable):
        return {'type': 'runnable', 'data': {'id': to_json_not_implemented(node.data)['id'], 'name': node.data.get_name()}}
    elif inspect.isclass(node.data) and issubclass(node.data, BaseModel):
        return {'type': 'schema', 'data': node.data.schema()} if with_schemas else {'type': 'schema', 'data': node_data_str(node)}
    else:
        return {'type': 'unknown', 'data': node_data_str(node)}