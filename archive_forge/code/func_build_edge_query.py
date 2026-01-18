import hashlib
import sys
from typing import Any, Dict, List, Optional, Union
from langchain_core.utils import get_from_env
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.graphs.graph_store import GraphStore
def build_edge_query(self, relationship: Relationship) -> str:
    source_query = f".has('id','{relationship.source.id}')"
    target_query = f".has('id','{relationship.target.id}')"
    base_query = f""""g.V(){source_query}.as('a')  \n            .V(){target_query}.as('b') \n            .choose(\n                __.inE('{relationship.type}').where(outV().as('a')),\n                __.identity(),\n                __.addE('{relationship.type}').from('a').to('b')\n            )        \n            """.replace('\n', '').replace('\t', '')
    for key, value in relationship.properties.items():
        base_query += f".property('{key}', '{value}')"
    return base_query