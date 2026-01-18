from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
def request_rerun(self, new_data: RerunData) -> bool:
    """Request that the ScriptRunner rerun its script.

        If the ScriptRunner has been stopped, this request can't be honored:
        return False.

        Otherwise, record the request and return True. The ScriptRunner will
        handle the rerun request as soon as it reaches an interrupt point.
        """
    with self._lock:
        if self._state == ScriptRequestType.STOP:
            return False
        if self._state == ScriptRequestType.CONTINUE:
            self._state = ScriptRequestType.RERUN
            self._rerun_data = new_data
            return True
        if self._state == ScriptRequestType.RERUN:
            coalesced_states = coalesce_widget_states(self._rerun_data.widget_states, new_data.widget_states)
            if new_data.fragment_id_queue:
                fragment_id_queue = [*self._rerun_data.fragment_id_queue]
                if (new_fragment_id := new_data.fragment_id_queue[0]) not in fragment_id_queue:
                    fragment_id_queue.append(new_fragment_id)
            else:
                fragment_id_queue = []
            self._rerun_data = RerunData(query_string=new_data.query_string, widget_states=coalesced_states, page_script_hash=new_data.page_script_hash, page_name=new_data.page_name, fragment_id_queue=fragment_id_queue)
            return True
        raise RuntimeError(f'Unrecognized ScriptRunnerState: {self._state}')