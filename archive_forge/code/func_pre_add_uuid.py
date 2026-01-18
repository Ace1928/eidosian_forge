from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
def pre_add_uuid(self, *args: Any, **kwargs) -> str:
    return to_uuid(self.configs, self.inputs, self.outputs, self._op_name, self._args, self._kwargs, args, kwargs)