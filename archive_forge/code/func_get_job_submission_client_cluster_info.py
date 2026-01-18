import dataclasses
import importlib
import logging
import json
import os
import yaml
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
from pkg_resources import packaging
import ray
import ssl
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.dashboard.modules.job.common import uri_to_http_components
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.utils import split_address
from ray.autoscaler._private.cli_logger import cli_logger
def get_job_submission_client_cluster_info(address: str, *, create_cluster_if_needed: Optional[bool]=False, cookies: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None, _use_tls: Optional[bool]=False) -> ClusterInfo:
    """Get address, cookies, and metadata used for SubmissionClient.

    If no port is specified in `address`, the Ray dashboard default will be
    inserted.

    Args:
        address: Address without the module prefix that is passed
            to SubmissionClient.
        create_cluster_if_needed: Indicates whether the cluster
            of the address returned needs to be running. Ray doesn't
            start a cluster before interacting with jobs, but other
            implementations may do so.

    Returns:
        ClusterInfo object consisting of address, cookies, and metadata
        for SubmissionClient to use.
    """
    scheme = 'https' if _use_tls else 'http'
    return ClusterInfo(address=f'{scheme}://{address}', cookies=cookies, metadata=metadata, headers=headers)