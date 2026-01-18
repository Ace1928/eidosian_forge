from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def assert_none(self, *nodes: Any) -> None:
    for node in nodes:
        if isinstance(node, list):
            if len(node) > 0:
                self.not_support(f'{node} is not empty')
        elif node is not None:
            expr = self.to_str(node)
            self.not_support(f'{expr}')