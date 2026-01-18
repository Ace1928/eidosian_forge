import hashlib
import sys
from typing import Any, Dict, List, Optional, Union
from langchain_core.utils import get_from_env
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.graph_store import GraphStore
def build_vertex_query(self, node: Node) -> str:
    base_query = f"g.V().has('id','{node.id}').fold()" + f".coalesce(unfold(),addV('{node.type}')" + f".property('id','{node.id}')" + f".property('type','{node.type}')"
    for key, value in node.properties.items():
        base_query += f".property('{key}', '{value}')"
    return base_query + ')'