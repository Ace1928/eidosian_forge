from __future__ import annotations
import importlib.util
import os
import stat
import typing
from email.utils import parsedate
import anyio
import anyio.to_thread
from starlette._utils import get_route_path
from starlette.datastructures import URL, Headers
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.types import Receive, Scope, Send
def get_directories(self, directory: PathLike | None=None, packages: list[str | tuple[str, str]] | None=None) -> list[PathLike]:
    """
        Given `directory` and `packages` arguments, return a list of all the
        directories that should be used for serving static files from.
        """
    directories = []
    if directory is not None:
        directories.append(directory)
    for package in packages or []:
        if isinstance(package, tuple):
            package, statics_dir = package
        else:
            statics_dir = 'statics'
        spec = importlib.util.find_spec(package)
        assert spec is not None, f'Package {package!r} could not be found.'
        assert spec.origin is not None, f'Package {package!r} could not be found.'
        package_directory = os.path.normpath(os.path.join(spec.origin, '..', statics_dir))
        assert os.path.isdir(package_directory), f"Directory '{statics_dir!r}' in package {package!r} could not be found."
        directories.append(package_directory)
    return directories