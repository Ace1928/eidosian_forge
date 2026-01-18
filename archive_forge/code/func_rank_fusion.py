import asyncio
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
from typing import (
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
def rank_fusion(self, query: str, run_manager: CallbackManagerForRetrieverRun, *, config: Optional[RunnableConfig]=None) -> List[Document]:
    """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
    retriever_docs = [retriever.invoke(query, patch_config(config, callbacks=run_manager.get_child(tag=f'retriever_{i + 1}'))) for i, retriever in enumerate(self.retrievers)]
    for i in range(len(retriever_docs)):
        retriever_docs[i] = [Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc for doc in retriever_docs[i]]
    fused_documents = self.weighted_reciprocal_rank(retriever_docs)
    return fused_documents