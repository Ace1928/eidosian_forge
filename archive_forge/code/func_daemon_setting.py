import json
import numbers
from collections import OrderedDict
from functools import update_wrapper
from pprint import pformat
from typing import Any
import click
from click import Context, ParamType
from kombu.utils.objects import cached_property
from celery._state import get_current_app
from celery.signals import user_preload_options
from celery.utils import text
from celery.utils.log import mlevel
from celery.utils.time import maybe_iso8601
def daemon_setting(self, ctx: Context, opt: CeleryOption, value: Any) -> Any:
    """
        Try to fetch deamonization option from applications settings.
        Use the daemon command name as prefix (eg. `worker` -> `worker_pidfile`)
        """
    return value or getattr(ctx.obj.app.conf, f'{ctx.command.name}_{self.name}', None)