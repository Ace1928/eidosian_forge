import fnmatch
import glob
import inspect
import io
import os
import pathlib
import shutil
import tarfile
import zipfile
import param
import requests
from bokeh.model import Model
from .config import config, panel_extension
from .io.resources import RESOURCE_URLS
from .reactive import ReactiveHTML
from .template.base import BasicTemplate
from .theme import Design
def bundle_resource_urls(verbose=False, external=True):
    for name, resource in RESOURCE_URLS.items():
        if verbose:
            print(f'Bundling shared resource {name}.')
        if 'zip' in resource:
            write_bundled_zip(name, resource)
        elif 'tar' in resource:
            write_bundled_tarball(resource, name=name)