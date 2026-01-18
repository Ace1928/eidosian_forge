import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def save_history(self):
    """This saves all cell executions in the current session as a new notebook."""
    try:
        from nbformat import v4, validator, write
    except ImportError:
        logger.error('Run pip install nbformat to save notebook history')
        return
    if self.shell is None:
        return
    cells = []
    hist = list(self.shell.history_manager.get_range(output=True))
    if len(hist) <= 1 or not self.settings.save_code:
        logger.info('not saving jupyter history')
        return
    try:
        for _, execution_count, exc in hist:
            if exc[1]:
                outputs = [v4.new_output(output_type='stream', name='stdout', text=exc[1])]
            else:
                outputs = []
            if self.outputs.get(execution_count):
                for out in self.outputs[execution_count]:
                    outputs.append(v4.new_output(output_type='display_data', data=out['data'], metadata=out['metadata'] or {}))
            cells.append(v4.new_code_cell(execution_count=execution_count, source=exc[0], outputs=outputs))
        if hasattr(self.shell, 'kernel'):
            language_info = self.shell.kernel.language_info
        else:
            language_info = {'name': 'python', 'version': sys.version}
        logger.info('saving %i cells to _session_history.ipynb', len(cells))
        nb = v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python %i' % sys.version_info[0], 'name': 'python%i' % sys.version_info[0], 'language': 'python'}, 'language_info': language_info})
        state_path = os.path.join('code', '_session_history.ipynb')
        wandb.run._set_config_wandb('session_history', state_path)
        filesystem.mkdir_exists_ok(os.path.join(wandb.run.dir, 'code'))
        with open(os.path.join(self.settings._tmp_code_dir, '_session_history.ipynb'), 'w', encoding='utf-8') as f:
            write(nb, f, version=4)
        with open(os.path.join(wandb.run.dir, state_path), 'w', encoding='utf-8') as f:
            write(nb, f, version=4)
    except (OSError, validator.NotebookValidationError) as e:
        logger.error('Unable to save ipython session history:\n%s', e)
        pass