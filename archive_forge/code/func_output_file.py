from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
def output_file(self, filename: PathLike, title: str='Bokeh Plot', mode: ResourcesMode | None=None, root_dir: PathLike | None=None) -> None:
    """ Configure output to a standalone HTML file.

        Calling ``output_file`` does not clear the effects of any other calls to
        |output_notebook|, etc. It adds an additional output destination
        (publishing to HTML files). Any other active output modes continue
        to be active.

        Args:
            filename (PathLike, e.g. str, Path) : a filename for saving the HTML document

            title (str, optional) : a title for the HTML document

            mode (str, optional) : how to include BokehJS (default: ``'cdn'``)

                One of: ``'inline'``, ``'cdn'``, ``'relative(-dev)'`` or
                ``'absolute(-dev)'``. See :class:`~bokeh.resources.Resources`
                for more details.

            root_dir (str, optional) : root dir to use for absolute resources
                (default: None)

                This value is ignored for other resource types, e.g. ``INLINE`` or ``CDN``.

        .. warning::
            The specified output file will be overwritten on every save, e.g.,
            every time ``show()`` or ``save()`` is called.

        """
    self._file = FileConfig(filename=filename, resources=Resources(mode=mode, root_dir=root_dir), title=title)
    if os.path.isfile(filename):
        log.info(f"Session output file '{filename}' already exists, will be overwritten.")