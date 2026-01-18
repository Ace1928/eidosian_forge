import logging
import re
from typing import List, Optional
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.llms import LlamaCpp
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
def search_tool(self, query: str, num_search_results: int=1) -> List[dict]:
    """Returns num_search_results pages per Google search."""
    query_clean = self.clean_search_query(query)
    result = self.search.results(query_clean, num_search_results)
    return result