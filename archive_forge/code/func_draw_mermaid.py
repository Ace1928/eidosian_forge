from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
from uuid import UUID, uuid4
from langchain_core.pydantic_v1 import BaseModel
def draw_mermaid(self, *, with_styles: bool=True, curve_style: CurveStyle=CurveStyle.LINEAR, node_colors: NodeColors=NodeColors(start='#ffdfba', end='#baffc9', other='#fad7de'), wrap_label_n_words: int=9) -> str:
    from langchain_core.runnables.graph_mermaid import draw_mermaid
    nodes = {node.id: node_data_str(node) for node in self.nodes.values()}
    first_node = self.first_node()
    first_label = node_data_str(first_node) if first_node is not None else None
    last_node = self.last_node()
    last_label = node_data_str(last_node) if last_node is not None else None
    return draw_mermaid(nodes=nodes, edges=self.edges, first_node_label=first_label, last_node_label=last_label, with_styles=with_styles, curve_style=curve_style, node_colors=node_colors, wrap_label_n_words=wrap_label_n_words)