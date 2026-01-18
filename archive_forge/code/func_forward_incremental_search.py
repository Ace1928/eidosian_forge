from __future__ import annotations
from prompt_toolkit import search
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition, control_is_searchable, is_searching
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from ..key_bindings import key_binding
@key_binding(filter=is_searching)
def forward_incremental_search(event: E) -> None:
    """
    Apply forward incremental search, but keep search buffer focused.
    """
    search.do_incremental_search(search.SearchDirection.FORWARD, count=event.arg)