from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Tuple, Union
from antlr4.tree.Tree import TerminalNode, Token, Tree
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from _qpd_antlr import QPDParser as qp
from qpd._parser.sql import QPDSql
from triad.utils.schema import unquote_name
from qpd.constants import AGGREGATION_FUNCTIONS, JOIN_TYPES
from qpd.dataframe import DataFrame, DataFrames
from qpd.workflow import QPDWorkflow
def add_join_relation(self, join: Join) -> None:
    if 'joins' not in self._data:
        self._data['joins'] = []
    for j in self._data['joins']:
        assert_or_throw(j.df_name != join.df_name, ValueError(f'{join.df_name} is already in join line'))
    self._data['joins'].append(join)