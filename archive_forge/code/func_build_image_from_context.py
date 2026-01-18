import os
from subprocess import PIPE, STDOUT, Popen
from typing import Optional, Union
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DOCKER_OPENJDK_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION
def build_image_from_context(context_dir: str, image_name: str):
    import docker
    client = docker.from_env()
    is_platform_supported = int(client.version()['Version'].split('.')[0]) >= 19
    platform_option = ['--platform', 'linux/amd64'] if is_platform_supported else []
    commands = ['docker', 'build', '-t', image_name, '-f', 'Dockerfile', *platform_option, '.']
    proc = Popen(commands, cwd=context_dir, stdout=PIPE, stderr=STDOUT, text=True, encoding='utf-8')
    for x in iter(proc.stdout.readline, ''):
        eprint(x, end='')
    if proc.wait():
        raise RuntimeError('Docker build failed.')