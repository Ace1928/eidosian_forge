import getpass
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import click
import requests
from wandb_gql import gql
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.lib import runid
from ...apis.internal import Api
def check_large_post() -> bool:
    print('Checking ability to send large payloads through proxy'.ljust(72, '.'), end='')
    descy = 'a' * int(10 ** 7)
    username = getpass.getuser()
    failed_test_strings = []
    query = gql('\n        query Project($entity: String!, $name: String!, $runName: String!, $desc: String!){\n            project(entityName: $entity, name: $name) {\n                run(name: $runName, desc: $desc) {\n                    name\n                    summaryMetrics\n                }\n            }\n        }\n        ')
    public_api = wandb.Api()
    client = public_api._base_client
    try:
        client._get_result(query, variable_values={'entity': username, 'name': PROJECT_NAME, 'runName': '', 'desc': descy}, timeout=60)
    except Exception as e:
        if isinstance(e, requests.HTTPError) and e.response is not None and (e.response.status_code == 413):
            failed_test_strings.append('Failed to send a large payload. Check nginx.ingress.kubernetes.io/proxy-body-size is "0".')
        else:
            failed_test_strings.append(f'Failed to send a large payload with error: {e}.')
    print_results(failed_test_strings, False)
    return len(failed_test_strings) == 0