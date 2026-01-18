from typing import Any, Dict, Optional
from ray._private.utils import split_address
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
def delete_applications(self):
    response = self._do_request('DELETE', DELETE_PATH)
    if response.status_code != 200:
        self._raise_error(response)