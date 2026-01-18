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
def ctx_to_col_names(self, ctx: Any) -> List[str]:
    expr = self.to_str(ctx, '')
    cols = self.dfs.name_resolver.get_cols(expr)
    if self.col_encoded:
        return [self._obj_to_col_name(x) for x in cols]
    else:
        return [x[0] + '.' + x[1] for x in cols]