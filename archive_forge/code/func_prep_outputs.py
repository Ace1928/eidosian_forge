import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.load.dump import dumpd
from langchain_core.memory import BaseMemory
from langchain_core.outputs import RunInfo
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from langchain_core.runnables import (
from langchain_core.runnables.utils import create_model
from langchain.schema import RUN_KEY
def prep_outputs(self, inputs: Dict[str, str], outputs: Dict[str, str], return_only_outputs: bool=False) -> Dict[str, str]:
    """Validate and prepare chain outputs, and save info about this run to memory.

        Args:
            inputs: Dictionary of chain inputs, including any inputs added by chain
                memory.
            outputs: Dictionary of initial chain outputs.
            return_only_outputs: Whether to only return the chain outputs. If False,
                inputs are also added to the final outputs.

        Returns:
            A dict of the final chain outputs.
        """
    self._validate_outputs(outputs)
    if self.memory is not None:
        self.memory.save_context(inputs, outputs)
    if return_only_outputs:
        return outputs
    else:
        return {**inputs, **outputs}