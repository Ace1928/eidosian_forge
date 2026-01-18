from __future__ import annotations
from logging import Logger
from typing import TYPE_CHECKING, Any, cast
from jinja2.exceptions import TemplateNotFound
from jupyter_server.base.handlers import FileFindHandler
@property
def serverapp(self) -> ServerApp:
    key = 'serverapp'
    return cast('ServerApp', self.settings[key])