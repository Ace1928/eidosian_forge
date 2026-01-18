import os
import re
import base64
import requests
import json
import functools
import contextlib
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any, TYPE_CHECKING
from lazyops.utils.logs import logger
from lazyops.types import BaseModel, lazyproperty, Literal
from pydantic.types import ByteSize
@lazyproperty
def is_pytorch_model(self):
    return bool(re_patterns['pytorch_model'].match(self.filename))