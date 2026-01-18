import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
def register_post_import_hook(hook: Union[str, Callable], hook_id: str, name: str) -> None:
    if isinstance(hook, (str,)):
        hook = _create_import_hook_from_string(hook)
    with _post_import_hooks_lock:
        global _post_import_hooks_init
        if not _post_import_hooks_init:
            _post_import_hooks_init = True
            sys.meta_path.insert(0, ImportHookFinder())
        module = sys.modules.get(name, None)
        if module is None:
            _post_import_hooks.setdefault(name, {}).update({hook_id: hook})
    if module is not None:
        hook(module)