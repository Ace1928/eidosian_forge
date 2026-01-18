import datetime
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar
import psutil
def publish_stats(self, stats: dict) -> None:
    ...