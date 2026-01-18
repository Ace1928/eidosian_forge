from __future__ import unicode_literals
from prompt_toolkit.completion import CompleteEvent, get_common_complete_suffix
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding.registry import Registry
import math
def display_completions_like_readline(event):
    """
    Key binding handler for readline-style tab completion.
    This is meant to be as similar as possible to the way how readline displays
    completions.

    Generate the completions immediately (blocking) and display them above the
    prompt in columns.

    Usage::

        # Call this handler when 'Tab' has been pressed.
        registry.add_binding(Keys.ControlI)(display_completions_like_readline)
    """
    b = event.current_buffer
    if b.completer is None:
        return
    complete_event = CompleteEvent(completion_requested=True)
    completions = list(b.completer.get_completions(b.document, complete_event))
    common_suffix = get_common_complete_suffix(b.document, completions)
    if len(completions) == 1:
        b.delete_before_cursor(-completions[0].start_position)
        b.insert_text(completions[0].text)
    elif common_suffix:
        b.insert_text(common_suffix)
    elif completions:
        _display_completions_like_readline(event.cli, completions)