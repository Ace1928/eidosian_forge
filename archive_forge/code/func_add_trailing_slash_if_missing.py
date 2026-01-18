import json
import logging
from typing import Dict, NamedTuple, Optional, Union
import urllib
from absl import flags
from utils import bq_consts
from utils import bq_error
def add_trailing_slash_if_missing(url: str) -> str:
    if not url.endswith('/'):
        return url + '/'
    return url