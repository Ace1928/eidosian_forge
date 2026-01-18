import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
def get_available_trackers():
    """Returns a list of all supported available trackers in the system"""
    return _available_trackers