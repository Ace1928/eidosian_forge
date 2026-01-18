from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def hooks_help_text(topic):
    segments = [_help_prefix]
    for hook_key in sorted(known_hooks.keys()):
        hooks = known_hooks_key_to_object(hook_key)
        segments.append(hooks.docs())
    return '\n'.join(segments)