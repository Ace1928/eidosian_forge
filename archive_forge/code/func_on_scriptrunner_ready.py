from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
def on_scriptrunner_ready(self) -> ScriptRequest:
    """Called by the ScriptRunner when it's about to run its script for
        the first time, and also after its script has successfully completed.

        If we have a RERUN request, return the request and set
        our internal state to CONTINUE.

        If we have a STOP request or no request, set our internal state
        to STOP.
        """
    with self._lock:
        if self._state == ScriptRequestType.RERUN:
            self._state = ScriptRequestType.CONTINUE
            return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)
        self._state = ScriptRequestType.STOP
        return ScriptRequest(ScriptRequestType.STOP)