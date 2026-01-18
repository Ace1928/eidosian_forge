from __future__ import annotations
import os
from pathlib import PurePath
import logging  # isort:skip
from bokeh.resources import Resources
from bokeh.settings import settings
def get_sphinx_resources(include_bokehjs_api=False):
    docs_cdn = settings.docs_cdn()
    if docs_cdn is None:
        resources = Resources(mode='cdn')
    elif docs_cdn == 'local':
        resources = Resources(mode='server', root_url='/en/latest/')
    elif docs_cdn.startswith('test:'):
        version = docs_cdn.split(':')[1]
        resources = Resources(mode='server', root_url=f'/en/{version}/')
    else:
        resources = Resources(mode='cdn', version=docs_cdn)
    if include_bokehjs_api:
        resources.components.append('bokeh-api')
    return resources