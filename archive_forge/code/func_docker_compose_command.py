import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, cast
from langsmith import env as ls_env
from langsmith import utils as ls_utils
@property
def docker_compose_command(self) -> List[str]:
    return ls_utils.get_docker_compose_command()