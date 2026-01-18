import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Any, Dict, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def jsonf(self, data: Dict[str, Any], data_dir: str, filename: str, is_output: Optional[bool]=True) -> None:
    """To log the input data as json file artifact."""
    file_path = os.path.join(data_dir, f'{filename}.json')
    save_json(data, file_path)
    self.run.log_file(file_path, name=filename, is_output=is_output)