import dataclasses
import json
import logging
import os
import platform
from typing import Any, Dict, Optional
import torch
@property
def hip_version(self) -> Optional[int]:
    return self.metadata['version']['hip']