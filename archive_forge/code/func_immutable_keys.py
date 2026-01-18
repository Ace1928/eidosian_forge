import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def immutable_keys() -> List[str]:
    """These are env keys that shouldn't change within a single process.

    We use this to maintain certain values between multiple calls to wandb.init within a single process.
    """
    return [DIR, ENTITY, PROJECT, API_KEY, IGNORE, DISABLE_CODE, DISABLE_GIT, DOCKER, MODE, BASE_URL, ERROR_REPORTING, CRASH_NOSYNC_TIME, MAGIC, USERNAME, USER_EMAIL, DIR, SILENT, CONFIG_PATHS, ANONYMOUS, RUN_GROUP, JOB_TYPE, TAGS, RESUME, AGENT_REPORT_INTERVAL, HTTP_TIMEOUT, HOST, DATA_DIR, ARTIFACT_DIR, ARTIFACT_FETCH_FILE_URL_BATCH_SIZE, CACHE_DIR, USE_V1_ARTIFACTS, DISABLE_SSL]