from typing import Any, Dict, Optional
from ray._private.utils import split_address
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
def deploy_applications(self, config: Dict):
    """Deploy multiple applications."""
    response = self._do_request('PUT', DEPLOY_PATH, json_data=config)
    if response.status_code != 200:
        self._raise_error(response)