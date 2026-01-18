from __future__ import annotations
from io import IOBase
from lazyops.types import BaseModel, Field
from lazyops.utils import logger
from typing import Optional, Dict, Any, List, Union, Sequence, Callable, TYPE_CHECKING
from .types import SlackContext, SlackPayload
from .configs import SlackSettings
def join_channel(self, channel_id: str, **kwargs):
    """
        Join a channel
        """
    if self.disabled:
        return
    if self.verbose:
        self.logger.info(f'Joining channel {channel_id}')
    self.sapi.conversations_join(channel=channel_id, **kwargs)