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
def download_ssh_key_aws():
    """Download the ssh key from the S3 bucket to the local machine."""
    print('======================================')
    print('Downloading ssh key...')
    s3_client = boto3.client('s3', region_name='us-west-2')
    bucket_name = 'aws-cluster-launcher-test'
    key_name = 'ray-autoscaler_59_us-west-2.pem'
    local_key_path = os.path.expanduser(f'~/.ssh/{key_name}')
    if not os.path.exists(os.path.dirname(local_key_path)):
        os.makedirs(os.path.dirname(local_key_path))
    s3_client.download_file(bucket_name, key_name, local_key_path)
    os.chmod(local_key_path, 256)