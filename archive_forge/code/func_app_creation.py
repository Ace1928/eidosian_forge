import base64
import json
import logging
import subprocess
import textwrap
import time
from typing import Any, Dict, List, Mapping, Optional
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
def app_creation(self) -> None:
    """Creates a Python file which will contain your Beam app definition."""
    script = textwrap.dedent('        import beam\n\n        # The environment your code will run on\n        app = beam.App(\n            name="{name}",\n            cpu={cpu},\n            memory="{memory}",\n            gpu="{gpu}",\n            python_version="{python_version}",\n            python_packages={python_packages},\n        )\n\n        app.Trigger.RestAPI(\n            inputs={{"prompt": beam.Types.String(), "max_length": beam.Types.String()}},\n            outputs={{"text": beam.Types.String()}},\n            handler="run.py:beam_langchain",\n        )\n\n        ')
    script_name = 'app.py'
    with open(script_name, 'w') as file:
        file.write(script.format(name=self.name, cpu=self.cpu, memory=self.memory, gpu=self.gpu, python_version=self.python_version, python_packages=self.python_packages))