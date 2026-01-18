import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
@property
def mins(self):
    return self.secs / 60