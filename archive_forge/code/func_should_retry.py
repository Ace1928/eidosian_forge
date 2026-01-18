import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
def should_retry() -> bool:
    if result.status_code == 429 and obey_rate_limit:
        return True
    if not retry_transient_errors:
        return False
    if result.status_code in gitlab.const.RETRYABLE_TRANSIENT_ERROR_CODES:
        return True
    if result.status_code == 409 and 'Resource lock' in result.reason:
        return True
    return False