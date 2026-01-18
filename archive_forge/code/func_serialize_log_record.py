import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
def serialize_log_record(self, log_record: Dict[str, Any]) -> str:
    """Returns the final representation of the log record."""
    return '%s%s' % (self.prefix, self.jsonify_log_record(log_record))