from __future__ import absolute_import
from flask import make_response, current_app
from flask_restful.utils import PY3
from json import dumps
def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""
    settings = current_app.config.get('RESTFUL_JSON', {})
    if current_app.debug:
        settings.setdefault('indent', 4)
        settings.setdefault('sort_keys', not PY3)
    dumped = dumps(data, **settings) + '\n'
    resp = make_response(dumped, code)
    resp.headers.extend(headers or {})
    return resp