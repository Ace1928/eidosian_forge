import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def get_summaries_as_docs(self, query: str) -> List[Document]:
    """
        Performs an arxiv search and returns list of
        documents, with summaries as the content.

        If an error occurs or no documents found, error text
        is returned instead. Wrapper for
        https://lukasschwab.me/arxiv.py/index.html#Search

        Args:
            query: a plaintext search query
        """
    try:
        if self.is_arxiv_identifier(query):
            results = self.arxiv_search(id_list=query.split(), max_results=self.top_k_results).results()
        else:
            results = self.arxiv_search(query[:self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results).results()
    except self.arxiv_exceptions as ex:
        return [Document(page_content=f'Arxiv exception: {ex}')]
    docs = [Document(page_content=result.summary, metadata={'Entry ID': result.entry_id, 'Published': result.updated.date(), 'Title': result.title, 'Authors': ', '.join((a.name for a in result.authors))}) for result in results]
    return docs