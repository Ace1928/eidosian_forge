from __future__ import annotations
import textwrap
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Mapping
from typing_extensions import TypeAlias
from streamlit.errors import DuplicateWidgetID
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.state.common import (
from streamlit.type_util import ValueFieldName
def coalesce_widget_states(old_states: WidgetStates | None, new_states: WidgetStates | None) -> WidgetStates | None:
    """Coalesce an older WidgetStates into a newer one, and return a new
    WidgetStates containing the result.

    For most widget values, we just take the latest version.

    However, any trigger_values (which are set by buttons) that are True in
    `old_states` will be set to True in the coalesced result, so that button
    presses don't go missing.
    """
    if not old_states and (not new_states):
        return None
    elif not old_states:
        return new_states
    elif not new_states:
        return old_states
    states_by_id: dict[str, WidgetState] = {wstate.id: wstate for wstate in new_states.widgets}
    trigger_value_types = [('trigger_value', False), ('string_trigger_value', None)]
    for old_state in old_states.widgets:
        for trigger_value_type, unset_value in trigger_value_types:
            if old_state.WhichOneof('value') == trigger_value_type and old_state.trigger_value != unset_value:
                new_trigger_val = states_by_id.get(old_state.id)
                if new_trigger_val and new_trigger_val.WhichOneof('value') == trigger_value_type:
                    states_by_id[old_state.id] = old_state
    coalesced = WidgetStates()
    coalesced.widgets.extend(states_by_id.values())
    return coalesced