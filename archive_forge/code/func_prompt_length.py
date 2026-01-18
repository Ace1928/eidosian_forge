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
def prompt_length(self, docs: List[Document], **kwargs: Any) -> Optional[int]:
    """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: List[Document], a list of documents to use to calculate the
                total prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
    inputs = self._get_inputs(docs, **kwargs)
    prompt = self.llm_chain.prompt.format(**inputs)
    return self.llm_chain._get_num_tokens(prompt)