from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
def assemble_output(self, *args: Any) -> None:
    task = self.add('assemble_df', *args)
    deps: ParamDict = {}
    self._add_dep(deps, task)
    self._add_task(Output(), deps)