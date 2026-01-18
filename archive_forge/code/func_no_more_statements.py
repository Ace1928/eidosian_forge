from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def no_more_statements(self):
    if self.rules and (not self.rules[0].is_consumed):
        self.rules[0].no_more_statements()
    elif self.rules:
        super().no_more_statements()