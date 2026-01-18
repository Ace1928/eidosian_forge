from __future__ import annotations
import glob
import json
import os
from typing import Any
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler
from ...utils import url_path_join, url_unescape
def kernelspec_model(handler, name, spec_dict, resource_dir):
    """Load a KernelSpec by name and return the REST API model"""
    d = {'name': name, 'spec': spec_dict, 'resources': {}}
    for resource in ['kernel.js', 'kernel.css']:
        if os.path.exists(pjoin(resource_dir, resource)):
            d['resources'][resource] = url_path_join(handler.base_url, 'kernelspecs', name, resource)
    for logo_file in glob.glob(pjoin(resource_dir, 'logo-*')):
        fname = os.path.basename(logo_file)
        no_ext, _ = os.path.splitext(fname)
        d['resources'][no_ext] = url_path_join(handler.base_url, 'kernelspecs', name, fname)
    return d