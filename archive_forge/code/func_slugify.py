from typing import Any, List, Optional
import re
import unicodedata
import ray
from ray.workflow.common import WORKFLOW_OPTIONS
from ray.dag import DAGNode, FunctionNode, InputNode
from ray.dag.input_node import InputAttributeNode, DAGInputData
from ray import cloudpickle
from ray._private import signature
from ray._private.client_mode_hook import client_mode_should_convert
from ray.workflow import serialization_context
from ray.workflow.common import (
from ray.workflow import workflow_context
from ray.workflow.workflow_state import WorkflowExecutionState, Task
def slugify(value: str, allow_unicode=False) -> str:
    """Adopted from
    https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, dots or hyphens. Also strip leading and
    trailing whitespace.
    """
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub('[^\\w.\\-]', '', value).strip()
    return re.sub('[-\\s]+', '-', value)