from abc import abstractmethod
from typing import Any, Dict, List
from langchain_community.graphs.graph_document import GraphDocument
@property
@abstractmethod
def get_structured_schema(self) -> Dict[str, Any]:
    """Returns the schema of the Graph database"""
    pass