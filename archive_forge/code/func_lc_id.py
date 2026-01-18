from abc import ABC
from typing import (
from typing_extensions import NotRequired
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
@classmethod
def lc_id(cls) -> List[str]:
    """A unique identifier for this class for serialization purposes.

        The unique identifier is a list of strings that describes the path
        to the object.
        """
    return [*cls.get_lc_namespace(), cls.__name__]