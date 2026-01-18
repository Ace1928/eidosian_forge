import atexit
import functools
import os
import pathlib
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union
from urllib.parse import quote
import sentry_sdk  # type: ignore
import sentry_sdk.utils  # type: ignore
import wandb
import wandb.env
import wandb.util
@_safe_noop
def configure_scope(self, tags: Optional[Dict[str, Any]]=None, process_context: Optional[str]=None) -> None:
    """Configure the Sentry scope for the current thread.

        This function should be called at the beginning of every thread that
        will send events to Sentry. It sets the tags that will be applied to
        all events sent from this thread. It also tries to start a session
        if one doesn't already exist for this thread.
        """
    assert self.hub is not None
    settings_tags = ('entity', 'project', 'run_id', 'run_url', 'sweep_url', 'sweep_id', 'deployment', '_disable_service', '_require_core', 'launch')
    with self.hub.configure_scope() as scope:
        scope.set_tag('platform', wandb.util.get_platform_name())
        if process_context:
            scope.set_tag('process_context', process_context)
        if tags is None:
            return None
        for tag in settings_tags:
            val = tags.get(tag, None)
            if val not in (None, ''):
                scope.set_tag(tag, val)
        if tags.get('_colab', None):
            python_runtime = 'colab'
        elif tags.get('_jupyter', None):
            python_runtime = 'jupyter'
        elif tags.get('_ipython', None):
            python_runtime = 'ipython'
        else:
            python_runtime = 'python'
        scope.set_tag('python_runtime', python_runtime)
        for obj in ('run', 'sweep'):
            obj_id, obj_url = (f'{obj}_id', f'{obj}_url')
            if tags.get(obj_url, None):
                continue
            try:
                app_url = wandb.util.app_url(tags['base_url'])
                entity, project = (quote(tags[k]) for k in ('entity', 'project'))
                scope.set_tag(obj_url, f'{app_url}/{entity}/{project}/{obj}s/{tags[obj_id]}')
            except Exception:
                pass
        email = tags.get('email')
        if email:
            scope.user = {'email': email}
    self.start_session()