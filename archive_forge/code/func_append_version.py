from __future__ import annotations
import logging # isort:skip
from tornado.web import StaticFileHandler
from bokeh.settings import settings
@classmethod
def append_version(cls, path: str) -> str:
    if settings.dev:
        return path
    else:
        version = StaticFileHandler.get_version(dict(static_path=settings.bokehjs_path()), path)
        return f'{path}?v={version}'