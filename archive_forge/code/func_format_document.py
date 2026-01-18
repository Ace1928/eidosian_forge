from __future__ import annotations
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
import yaml
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import (
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables.utils import create_model
def format_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. `page_content`:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.prompts import PromptTemplate

            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
    """
    return prompt.format(**_get_document_info(doc, prompt))