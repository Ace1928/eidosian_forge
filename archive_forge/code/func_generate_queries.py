import asyncio
import logging
from typing import List, Optional, Sequence
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.chains.llm import LLMChain
def generate_queries(self, question: str, run_manager: CallbackManagerForRetrieverRun) -> List[str]:
    """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
    response = self.llm_chain({'question': question}, callbacks=run_manager.get_child())
    lines = response['text']
    if self.verbose:
        logger.info(f'Generated queries: {lines}')
    return lines