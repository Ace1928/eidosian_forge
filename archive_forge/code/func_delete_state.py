from __future__ import annotations
import datetime
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator
def delete_state(self, session_id: str, expired_only: bool=False):
    if session_id not in self.session_data:
        return
    to_delete = []
    session_state = self.session_data[session_id]
    for component, value, expired in session_state.state_components:
        if not expired_only or expired:
            component.delete_callback(value)
            to_delete.append(component._id)
    for component in to_delete:
        del session_state._data[component]