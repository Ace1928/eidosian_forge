from __future__ import annotations
import dataclasses
import typing as T
from .. import mparser
def plusassign(self, value: mparser.BaseNode, varname: str) -> mparser.PlusAssignmentNode:
    """Create a "+=" node

        :param value: The value to add
        :param varname: The variable to assign
        :return: The PlusAssignmentNode
        """
    return mparser.PlusAssignmentNode(self.identifier(varname), self._symbol('+='), value)