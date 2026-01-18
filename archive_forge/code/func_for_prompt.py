from __future__ import unicode_literals
from .defaults import load_key_bindings
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.key_binding.registry import Registry, ConditionalRegistry, MergedRegistry
@classmethod
def for_prompt(cls, **kw):
    """
        Create a ``KeyBindingManager`` with the defaults for an input prompt.
        This activates the key bindings for abort/exit (Ctrl-C/Ctrl-D),
        incremental search and auto suggestions.

        (Not for full screen applications.)
        """
    kw.setdefault('enable_abort_and_exit_bindings', True)
    kw.setdefault('enable_search', True)
    kw.setdefault('enable_auto_suggest_bindings', True)
    return cls(**kw)