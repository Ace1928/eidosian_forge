from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def last_node(self) -> Optional[Node]:
    """Find the single node that is not a source of any edge.
        If there is no such node, or there are multiple, return None.
        When drawing the graph this node would be the destination.
        """
    sources = {edge.source for edge in self.edges}
    found: List[Node] = []
    for node in self.nodes.values():
        if node.id not in sources:
            found.append(node)
    return found[0] if len(found) == 1 else None