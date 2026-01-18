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
def bootstrap_gcp(config):
    config = copy.deepcopy(config)
    check_legacy_fields(config)
    config['head_node'] = {}
    if _has_tpus_in_node_configs(config):
        config['provider'][HAS_TPU_PROVIDER_FIELD] = True
    crm, iam, compute, tpu = construct_clients_from_provider_config(config['provider'])
    config = _configure_project(config, crm)
    config = _configure_iam_role(config, crm, iam)
    config = _configure_key_pair(config, compute)
    config = _configure_subnet(config, compute)
    return config