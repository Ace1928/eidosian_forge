from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def node_data_str(node: Node) -> str:
    from langchain_core.runnables.base import Runnable
    if not is_uuid(node.id):
        return node.id
    elif isinstance(node.data, Runnable):
        try:
            data = str(node.data)
            if data.startswith('<') or data[0] != data[0].upper() or len(data.splitlines()) > 1:
                data = node.data.__class__.__name__
            elif len(data) > 42:
                data = data[:42] + '...'
        except Exception:
            data = node.data.__class__.__name__
    else:
        data = node.data.__name__
    return data if not data.startswith('Runnable') else data[8:]