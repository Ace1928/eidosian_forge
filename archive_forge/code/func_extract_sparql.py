from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_community.graphs import NeptuneRdfGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import SPARQL_QA_PROMPT
from langchain.chains.llm import LLMChain
def extract_sparql(query: str) -> str:
    query = query.strip()
    querytoks = query.split('```')
    if len(querytoks) == 3:
        query = querytoks[1]
        if query.startswith('sparql'):
            query = query[6:]
    elif query.startswith('<sparql>') and query.endswith('</sparql>'):
        query = query[8:-9]
    return query