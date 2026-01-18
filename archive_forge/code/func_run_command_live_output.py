import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def run_command_live_output(args: List[Any]) -> str:
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1) as process:
        stdout = ''
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            index = chunk.find(b'\r')
            if index != -1:
                print(chunk.decode(), end='')
            else:
                stdout += chunk.decode()
                print(chunk.decode(), end='\r')
        print(stdout)
    return_code = process.wait()
    if return_code != 0:
        raise DockerError(args, return_code, stdout.encode())
    return stdout