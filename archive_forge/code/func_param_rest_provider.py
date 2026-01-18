import json
import os
import tempfile
import traceback
from runpy import run_path
from unittest.mock import MagicMock
from urllib.parse import parse_qs
import param
from tornado import web
from tornado.wsgi import WSGIContainer
from ..entry_points import entry_points_for
from .state import state
def param_rest_provider(files, endpoint):
    """
    Returns a Param based REST API given the scripts or notebooks
    containing the tranquilized functions.

    Arguments
    ---------
    files: list(str)
      A list of paths being served
    endpoint: str
      The endpoint to serve the REST API on

    Returns
    -------
    A Tornado routing pattern containing the route and handler
    """
    for filename in files:
        extension = filename.split('.')[-1]
        if extension == 'py':
            try:
                run_path(filename)
            except Exception:
                param.main.param.warning('Could not run app script on REST server startup.')
        elif extension == 'ipynb':
            try:
                import nbconvert
            except ImportError:
                raise ImportError('Please install nbconvert to serve Jupyter Notebooks.') from None
            from nbconvert import ScriptExporter
            exporter = ScriptExporter()
            source, _ = exporter.from_filename(filename)
            source_dir = os.path.dirname(filename)
            with tempfile.NamedTemporaryFile(mode='w', dir=source_dir, delete=True) as tmp:
                tmp.write(source)
                tmp.flush()
                try:
                    run_path(tmp.name, init_globals={'get_ipython': MagicMock()})
                except Exception:
                    param.main.param.warning('Could not run app notebook on REST server startup.')
        else:
            raise ValueError('{} is not a script (.py) or notebook (.ipynb)'.format(filename))
    if endpoint and (not endpoint.endswith('/')):
        endpoint += '/'
    return [('^/%s.*' % endpoint if endpoint else '^.*', ParamHandler, dict(root=endpoint))]