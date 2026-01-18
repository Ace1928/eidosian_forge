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
@property
def config_specs(self) -> List[ConfigurableFieldSpec]:
    """List configurable fields for this runnable."""
    return get_unique_config_specs((spec for retriever in self.retrievers for spec in retriever.config_specs))