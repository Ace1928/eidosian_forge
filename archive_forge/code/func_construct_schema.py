from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain_community.graphs.graph_store import GraphStore
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
def construct_schema(structured_schema: Dict[str, Any], include_types: List[str], exclude_types: List[str]) -> str:
    """Filter the schema based on included or excluded types"""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types
    filtered_schema: Dict[str, Any] = {'node_props': {k: v for k, v in structured_schema.get('node_props', {}).items() if filter_func(k)}, 'rel_props': {k: v for k, v in structured_schema.get('rel_props', {}).items() if filter_func(k)}, 'relationships': [r for r in structured_schema.get('relationships', []) if all((filter_func(r[t]) for t in ['start', 'end', 'type']))]}
    formatted_node_props = []
    for label, properties in filtered_schema['node_props'].items():
        props_str = ', '.join([f'{prop['property']}: {prop['type']}' for prop in properties])
        formatted_node_props.append(f'{label} {{{props_str}}}')
    formatted_rel_props = []
    for rel_type, properties in filtered_schema['rel_props'].items():
        props_str = ', '.join([f'{prop['property']}: {prop['type']}' for prop in properties])
        formatted_rel_props.append(f'{rel_type} {{{props_str}}}')
    formatted_rels = [f'(:{el['start']})-[:{el['type']}]->(:{el['end']})' for el in filtered_schema['relationships']]
    return '\n'.join(['Node properties are the following:', ','.join(formatted_node_props), 'Relationship properties are the following:', ','.join(formatted_rel_props), 'The relationships are the following:', ','.join(formatted_rels)])