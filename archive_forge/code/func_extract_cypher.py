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
def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    pattern = '```(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else text