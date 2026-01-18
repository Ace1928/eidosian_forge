import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
def jsonify_log_record(self, log_record):
    """Returns a json string of the log record."""
    return self.json_serializer(log_record, default=self.json_default, cls=self.json_encoder, indent=self.json_indent, ensure_ascii=self.json_ensure_ascii)