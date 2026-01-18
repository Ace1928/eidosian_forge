import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
def get_score_attribute(self) -> str:
    """Document Score Confidence"""
    if self.ScoreAttributes is not None:
        return self.ScoreAttributes['ScoreConfidence']
    else:
        return 'NOT_AVAILABLE'