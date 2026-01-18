import logging
import os
import random
import string
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.utils import get_from_dict_or_env
from langchain_community.callbacks.utils import (
def langchain_artifact(self, chain: Any) -> None:
    self.mlflow.langchain.log_model(chain, 'langchain-model', run_id=self.run_id)