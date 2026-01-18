import html
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar
from triad import ParamDict, SerializableRLock, assert_or_throw
from .._utils.registry import fugue_plugin
from ..exceptions import FugueDatasetEmptyError
def repr_html(self) -> str:
    """The HTML representation of the :class:`~.Dataset`

        :return: the HTML representation
        """
    return html.escape(self.repr())