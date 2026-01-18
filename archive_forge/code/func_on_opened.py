from __future__ import annotations
import logging
import os.path
import re
from dataclasses import dataclass, field
from typing import Optional
from watchdog.utils.patterns import match_any_paths
def on_opened(self, event: FileSystemEvent) -> None:
    super().on_opened(event)
    self.logger.info('Opened file: %s', event.src_path)