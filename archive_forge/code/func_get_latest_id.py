import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pytz
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def get_latest_id(self):
    return self.autolog_id