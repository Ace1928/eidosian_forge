from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.combine_documents.base import (
from langchain.chains.llm import LLMChain
def format_docs(inputs: dict) -> str:
    return document_separator.join((format_document(doc, _document_prompt) for doc in inputs[DOCUMENTS_KEY]))