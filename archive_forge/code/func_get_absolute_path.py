from __future__ import annotations
import logging # isort:skip
import os
from pathlib import Path
from tornado.web import HTTPError, StaticFileHandler
from ...core.types import PathLike
@classmethod
def get_absolute_path(cls, root: dict[str, PathLike], path: str) -> str:
    try:
        name, artifact_path = path.split(os.sep, 1)
    except ValueError:
        raise HTTPError(404)
    artifacts_dir = root.get(name, None)
    if artifacts_dir is not None:
        return super().get_absolute_path(str(artifacts_dir), artifact_path)
    else:
        raise HTTPError(404)