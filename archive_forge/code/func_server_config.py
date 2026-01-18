from __future__ import annotations
from logging import Logger
from typing import TYPE_CHECKING, Any, cast
from jinja2.exceptions import TemplateNotFound
from jupyter_server.base.handlers import FileFindHandler
@property
def server_config(self) -> Config:
    return cast('Config', self.settings['config'])