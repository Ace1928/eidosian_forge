import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
def log_compilation_event(metrics):
    log.info('%s', metrics)