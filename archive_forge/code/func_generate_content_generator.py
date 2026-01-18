import os
import sys
import traceback
from typing import Generator, Tuple, Union, List
import nbformat
import tornado.web
from jupyter_server.config_manager import recursive_update
from nbclient.exceptions import CellExecutionError
from nbclient.util import ensure_async
from nbconvert.preprocessors import ClearOutputPreprocessor
from traitlets.config.configurable import LoggingConfigurable
from .execute import VoilaExecutor, strip_code_cell_warnings
from .exporter import VoilaExporter
from .paths import collect_template_paths
from .utils import ENV_VARIABLE
def generate_content_generator(self, kernel_id: Union[str, None]=None, kernel_future=None) -> Generator:

    async def inner_kernel_start(nb):
        return await self._jinja_kernel_start(nb, kernel_id, kernel_future)

    def inner_cell_generator(nb, kernel_id):
        return self._jinja_cell_generator(nb, kernel_id)
    extra_context = {'kernel_start': inner_kernel_start, 'cell_generator': inner_cell_generator, 'notebook_execute': self._jinja_notebook_execute}
    return self.exporter.generate_from_notebook_node(self.notebook, resources=self.resources, extra_context=extra_context)