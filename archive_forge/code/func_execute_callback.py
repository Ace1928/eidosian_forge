from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from ray.autoscaler._private.cli_logger import cli_logger
def execute_callback(self, event: CreateClusterEvent, event_data: Optional[Dict[str, Any]]=None):
    """Executes all callbacks for event.

        Args:
            event: Event that is invoked. See CreateClusterEvent
                for details on the available events.
            event_data (Dict[str, Any]): Argument that is passed to each
                callable object stored for this particular event.
        """
    if event_data is None:
        event_data = {}
    event_data['event_name'] = event
    if event in self.callback_map:
        for callback in self.callback_map[event]:
            callback(event_data)