import asyncio
import logging
import os
from typing import TYPE_CHECKING, Optional
import wandb
from wandb.sdk.lib.paths import LogicalPath
def progress(self, total_bytes: int) -> None:
    self._stats.update_uploaded_file(self.save_name, total_bytes)