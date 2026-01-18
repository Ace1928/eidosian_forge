import datetime
import decimal
from io import BytesIO
import os
import time
from typing import Dict, Union
import uuid
from .const import (
def prepare_uuid(data, schema):
    """Converts uuid.UUID to
    string formatted UUID xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    """
    if isinstance(data, uuid.UUID):
        return str(data)
    else:
        return data