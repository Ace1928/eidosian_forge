from __future__ import annotations
import os
import typing as t
from datetime import timedelta
from .cli import AppGroup
from .globals import current_app
from .helpers import send_from_directory
from .sansio.blueprints import Blueprint as SansioBlueprint
from .sansio.blueprints import BlueprintSetupState as BlueprintSetupState  # noqa
from .sansio.scaffold import _sentinel
def send_static_file(self, filename: str) -> Response:
    """The view function used to serve files from
        :attr:`static_folder`. A route is automatically registered for
        this view at :attr:`static_url_path` if :attr:`static_folder` is
        set.

        Note this is a duplicate of the same method in the Flask
        class.

        .. versionadded:: 0.5

        """
    if not self.has_static_folder:
        raise RuntimeError("'static_folder' must be set to serve static_files.")
    max_age = self.get_send_file_max_age(filename)
    return send_from_directory(t.cast(str, self.static_folder), filename, max_age=max_age)