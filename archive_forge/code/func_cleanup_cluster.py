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
def cleanup_cluster(cluster_config):
    """
    Clean up the cluster using the given cluster configuration file.

    Args:
        cluster_config: The path of the cluster configuration file.
    """
    print('======================================')
    print('Cleaning up cluster...')
    last_error = None
    num_tries = 3
    for i in range(num_tries):
        try:
            subprocess.run(['ray', 'down', '-v', '-y', str(cluster_config)], check=True, capture_output=True)
            return
        except subprocess.CalledProcessError as e:
            print(f'ray down fails[{i + 1}/{num_tries}]: ')
            print(e.output.decode('utf-8'))
            traceback.print_exc()
            print(f'stdout:\n{e.stdout.decode('utf-8')}')
            print(f'stderr:\n{e.stderr.decode('utf-8')}')
            last_error = e
    raise last_error