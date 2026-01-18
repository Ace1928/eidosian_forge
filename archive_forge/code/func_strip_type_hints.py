import inspect
import itertools
import textwrap
from typing import Callable, List, Mapping, Optional
import wandb
from .wandb_logging import wandb_log
import typing
from typing import NamedTuple
import collections
from collections import namedtuple
import kfp
from kfp import components
from kfp.components import InputPath, OutputPath
import wandb
def strip_type_hints(source_code: str) -> str:
    """Strip type hints from source code.

    This function is modified from KFP.  The original source is below:
    https://github.com/kubeflow/pipelines/blob/b6406b02f45cdb195c7b99e2f6d22bf85b12268b/sdk/python/kfp/components/_python_op.py#L237-L248.
    """
    return source_code