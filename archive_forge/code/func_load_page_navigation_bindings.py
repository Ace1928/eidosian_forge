from __future__ import annotations
from prompt_toolkit.filters import buffer_has_focus, emacs_mode, vi_mode
from prompt_toolkit.key_binding.key_bindings import (
from .scroll import (
def load_page_navigation_bindings() -> KeyBindingsBase:
    """
    Load both the Vi and Emacs bindings for page navigation.
    """
    return ConditionalKeyBindings(merge_key_bindings([load_emacs_page_navigation_bindings(), load_vi_page_navigation_bindings()]), buffer_has_focus)