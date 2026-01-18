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
def build_tranquilize_application(files):
    from tranquilizer.handler import NotebookHandler, ScriptHandler
    from tranquilizer.main import UnsupportedFileType, make_app
    functions = []
    for filename in files:
        extension = filename.split('.')[-1]
        if extension == 'py':
            source = ScriptHandler(filename)
        elif extension == 'ipynb':
            try:
                import nbconvert
            except ImportError as e:
                raise ImportError('Please install nbconvert to serve Jupyter Notebooks.') from e
            source = NotebookHandler(filename)
        else:
            raise UnsupportedFileType('{} is not a script (.py) or notebook (.ipynb)'.format(filename))
        functions.extend(source.tranquilized_functions)
    return make_app(functions, 'Panel REST API', prefix='rest/')