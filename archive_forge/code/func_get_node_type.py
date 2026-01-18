import copy
import json
import logging
import os
import re
import time
from functools import partial, reduce
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from googleapiclient import discovery, errors
from ray._private.accelerators import TPUAcceleratorManager, tpu
from ray.autoscaler._private.gcp.node import MAX_POLLS, POLL_INTERVAL, GCPNodeType
from ray.autoscaler._private.util import check_legacy_fields
def get_node_type(node: dict) -> GCPNodeType:
    """Returns node type based on the keys in ``node``.

    This is a very simple check. If we have a ``machineType`` key,
    this is a Compute instance. If we don't have a ``machineType`` key,
    but we have ``acceleratorType``, this is a TPU. Otherwise, it's
    invalid and an exception is raised.

    This works for both node configs and API returned nodes.
    """
    if 'machineType' not in node and 'acceleratorType' not in node and ('acceleratorConfig' not in node):
        raise ValueError(f"Invalid node. For a Compute instance, 'machineType' is required.For a TPU instance, 'acceleratorType' OR 'acceleratorConfig' and no 'machineType' is required. Got {list(node)}.")
    if 'machineType' not in node and ('acceleratorType' in node or 'acceleratorConfig' in node):
        _validate_tpu_config(node)
        if not _is_single_host_tpu(node):
            logger.warning('TPU pod detected. Note that while the cluster launcher can create multiple TPU pods, proper autoscaling will not work as expected, as all hosts in a TPU pod need to execute the same program. Proceed with caution.')
        return GCPNodeType.TPU
    return GCPNodeType.COMPUTE