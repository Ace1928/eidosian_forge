from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
def output_notebook(self, notebook_type: NotebookType='jupyter') -> None:
    """ Generate output in notebook cells.

        Calling ``output_notebook`` does not clear the effects of any other
        calls to |output_file|, etc. It adds an additional output destination
        (publishing to notebook output cells). Any other active output modes
        continue to be active.

        Returns:
            None

        """
    self._notebook = True
    self.notebook_type = notebook_type