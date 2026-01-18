import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def get_langchain_env_var_metadata() -> dict:
    """Retrieve the langchain environment variables."""
    excluded = {'LANGCHAIN_API_KEY', 'LANGCHAIN_ENDPOINT', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_PROJECT', 'LANGCHAIN_SESSION'}
    langchain_metadata = {k: v for k, v in os.environ.items() if (k.startswith('LANGCHAIN_') or k.startswith('LANGSMITH_')) and k not in excluded and ('key' not in k.lower()) and ('secret' not in k.lower()) and ('token' not in k.lower())}
    env_revision_id = langchain_metadata.pop('LANGCHAIN_REVISION_ID', None)
    if env_revision_id:
        langchain_metadata['revision_id'] = env_revision_id
    elif (default_revision_id := _get_default_revision_id()):
        langchain_metadata['revision_id'] = default_revision_id
    return langchain_metadata