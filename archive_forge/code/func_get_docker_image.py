import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
import boto3
import yaml
from google.cloud import storage
import ray
def get_docker_image(docker_override):
    """
    Get the docker image to use for the head node and worker nodes.

    Args:
        docker_override: The value of the --docker-override flag.

    Returns:
        The docker image to use for the head node and worker nodes, or None if not
        applicable.
    """
    if docker_override == 'latest':
        return 'rayproject/ray:latest-py38'
    elif docker_override == 'nightly':
        return 'rayproject/ray:nightly-py38'
    elif docker_override == 'commit':
        if re.match('^[0-9]+.[0-9]+.[0-9]+$', ray.__version__):
            return f'rayproject/ray:{ray.__version__}.{ray.__commit__[:6]}-py38'
        else:
            print(f'Error: docker image is only available for release version, but we get: {ray.__version__}')
            sys.exit(1)
    return None