from __future__ import annotations
import typing as t
from .exceptions import ListparserError
from .xml_handler import XMLHandler
def expect_text(self, _: t.Any) -> None:
    """Flag that text content is anticipated.

        Many start_* methods only need to prepare for text content.
        This method exists so those start_* methods can be declared
        as aliases for this method.
        """
    self.flag_expect_text = True
    self.text = []