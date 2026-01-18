from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.helper import object_to_dict
@property
def byName(self):
    return self.copy(by_name=True)