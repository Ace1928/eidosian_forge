import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def default_dump_dir():
    return os.path.join(Path.home(), '.triton', 'dump')