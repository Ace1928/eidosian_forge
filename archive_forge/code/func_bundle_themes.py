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
def bundle_themes(verbose=False, external=True):
    for name, design in param.concrete_descendents(Design).items():
        if verbose:
            print(f'Bundling {name} design resources')
        if design._resources.get('bundle', True) and external:
            write_component_resources(name, design)
    theme_bundle_dir = BUNDLE_DIR / 'theme'
    theme_bundle_dir.mkdir(parents=True, exist_ok=True)
    for design_css in glob.glob(str(BASE_DIR / 'theme' / 'css' / '*.css')):
        shutil.copyfile(design_css, theme_bundle_dir / os.path.basename(design_css))