from __future__ import annotations
import logging
import os
import pathlib
import platform
from typing import Optional, Tuple
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.document_loaders.base import BaseLoader
def get_runtime() -> Tuple[Framework, Runtime]:
    """Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.
    """
    runtime_env = get_runtime_environment()
    framework = Framework(name='langchain', version=runtime_env.get('library_version', None))
    uname = platform.uname()
    runtime = Runtime(host=uname.node, path=os.environ['PWD'], platform=runtime_env.get('platform', 'unknown'), os=uname.system, os_version=uname.version, ip=get_ip(), language=runtime_env.get('runtime', 'unknown'), language_version=runtime_env.get('runtime_version', 'unknown'))
    if 'Darwin' in runtime.os:
        runtime.type = 'desktop'
        runtime.runtime = 'Mac OSX'
    logger.debug(f'framework {framework}')
    logger.debug(f'runtime {runtime}')
    return (framework, runtime)