from __future__ import annotations
from typing import TYPE_CHECKING
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.tools.gmail.utils import build_resource_service
@classmethod
def from_api_resource(cls, api_resource: Resource) -> 'GmailBaseTool':
    """Create a tool from an api resource.

        Args:
            api_resource: The api resource to use.

        Returns:
            A tool.
        """
    return cls(service=api_resource)