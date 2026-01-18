from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from ray.autoscaler._private.cli_logger import cli_logger
def clear_callbacks_for_event(self, event: str):
    """Clears stored callable objects for event.

        Args:
            event: Event that has callable objects stored in map.
                See CreateClusterEvent for details on the available events.
        """
    if event in self.callback_map:
        del self.callback_map[event]