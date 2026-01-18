from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def associate_batch(self, text_pairs: List[Tuple[str, str]]) -> None:
    """Given a batch of (source, target) pairs, the retriever associates
        each source phrase with the corresponding target phrase.

        Args:
            text_pairs: list of (source, target) text pairs. For each pair in
            this list, the source will be associated with the target.
        """
    self.db.associate_batch(text_pairs)