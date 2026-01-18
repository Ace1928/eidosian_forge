from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra
from langchain.chains.base import Chain
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.chains.llm import LLMChain
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Call the base embeddings."""
    return self.base_embeddings.embed_documents(texts)