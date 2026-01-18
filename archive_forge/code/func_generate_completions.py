from __future__ import unicode_literals
from prompt_toolkit.completion import CompleteEvent, get_common_complete_suffix
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding.registry import Registry
import math
def generate_completions(event):
    """
    Tab-completion: where the first tab completes the common suffix and the
    second tab lists all the completions.
    """
    b = event.current_buffer
    if b.complete_state:
        b.complete_next()
    else:
        event.cli.start_completion(insert_common_part=True, select_first=False)