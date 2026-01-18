import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
def signpost_event(category: str, name: str, parameters: Dict[str, Any]):
    log.info('%s %s: %r', category, name, parameters)