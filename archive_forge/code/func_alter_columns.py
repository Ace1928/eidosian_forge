import sys
from collections import defaultdict
from typing import (
from uuid import uuid4
from adagio.specs import WorkflowSpec
from triad import (
from fugue._utils.exception import modify_traceback
from fugue.collections.partition import PartitionSpec
from fugue.collections.sql import StructuredRawSQL
from fugue.collections.yielded import Yielded
from fugue.column import ColumnExpr
from fugue.column import SelectColumns as ColumnsSelect
from fugue.column import all_cols, col, lit
from fugue.constants import (
from fugue.dataframe import DataFrame, LocalBoundedDataFrame, YieldedDataFrame
from fugue.dataframe.api import is_df
from fugue.dataframe.dataframes import DataFrames
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowError
from fugue.execution.api import engine_context
from fugue.extensions._builtins import (
from fugue.extensions.transformer.convert import _to_output_transformer, _to_transformer
from fugue.rpc import to_rpc_handler
from fugue.rpc.base import EmptyRPCHandler
from fugue.workflow._checkpoint import StrongCheckpoint, WeakCheckpoint
from fugue.workflow._tasks import Create, FugueTask, Output, Process
from fugue.workflow._workflow_context import FugueWorkflowContext
def alter_columns(self: TDF, columns: Any) -> TDF:
    """Change column types

        :param columns: |SchemaLikeObject|
        :return: a new dataframe with the new column types
        :rtype: :class:`~.WorkflowDataFrame`

        .. note::

            The output dataframe will not change the order of original schema.

        .. admonition:: Examples

            >>> df.alter_columns("a:int,b;str")
        """
    df = self.workflow.process(self, using=AlterColumns, params=dict(columns=columns))
    return self._to_self_type(df)