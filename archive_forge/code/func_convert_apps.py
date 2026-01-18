from __future__ import annotations
import concurrent.futures
import dataclasses
import json
import os
import pathlib
import uuid
from typing import (
import bokeh
from bokeh.application.application import SessionContext
from bokeh.application.handlers.code import CodeHandler
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import FILE, MACROS, get_env
from bokeh.document import Document
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import RenderItem, standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.util.serialization import make_id
from .. import __version__, config
from ..util import base_version, escape
from .application import Application, build_single_handler_application
from .loading import LOADING_INDICATOR_CSS_CLASS
from .mime_render import find_requirements
from .resources import (
from .state import set_curdoc, state
import asyncio
from panel.io.pyodide import init_doc, write_doc
def convert_apps(apps: str | os.PathLike | List[str | os.PathLike], dest_path: str | os.PathLike | None=None, title: str | None=None, runtime: Runtimes='pyodide-worker', requirements: List[str] | Literal['auto'] | os.PathLike='auto', prerender: bool=True, build_index: bool=True, build_pwa: bool=True, pwa_config: Dict[Any, Any]={}, max_workers: int=4, panel_version: Literal['auto', 'local'] | str='auto', http_patch: bool=True, inline: bool=False, compiled: bool=False, verbose: bool=True):
    """
    Arguments
    ---------
    apps: str | List[str]
        The filename(s) of the Panel/Bokeh application(s) to convert.
    dest_path: str | pathlib.Path
        The directory to write the converted application(s) to.
    title: str | None
        A title for the application(s). Also used to generate unique
        name for the application cache to ensure.
    runtime: 'pyodide' | 'pyscript' | 'pyodide-worker'
        The runtime to use for running Python in the browser.
    requirements: 'auto' | List[str] | os.PathLike | Dict[str, 'auto' | List[str] | os.PathLike]
        The list of requirements to include (in addition to Panel).
        By default automatically infers dependencies from imports
        in the application. May also provide path to a requirements.txt
    prerender: bool
        Whether to pre-render the components so the page loads.
    build_index: bool
        Whether to write an index page (if there are multiple apps).
    build_pwa: bool
        Whether to write files to define a progressive web app (PWA) including
        a manifest and a service worker that caches the application locally
    pwa_config: Dict[Any, Any]
        Configuration for the PWA including (see https://developer.mozilla.org/en-US/docs/Web/Manifest)

          - display: Display options ('fullscreen', 'standalone', 'minimal-ui' 'browser')
          - orientation: Preferred orientation
          - background_color: The background color of the splash screen
          - theme_color: The theme color of the application
    max_workers: int
        The maximum number of parallel workers
    panel_version: 'auto' | 'local'] | str
'       The panel version to include.
    http_patch: bool
        Whether to patch the HTTP request stack with the pyodide-http library
        to allow urllib3 and requests to work.
    inline: bool
        Whether to inline resources.
    compiled: bool
        Whether to use the compiled and faster version of Pyodide.
    """
    if isinstance(apps, str):
        apps = [apps]
    if dest_path is None:
        dest_path = pathlib.Path('./')
    elif not isinstance(dest_path, pathlib.PurePath):
        dest_path = pathlib.Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    manifest = 'site.webmanifest' if build_pwa else None
    if isinstance(requirements, dict):
        app_requirements = {}
        for app in apps:
            matches = [deps for name, deps in requirements.items() if app.endswith(name.replace(os.path.sep, '/'))]
            app_requirements[app] = matches[0] if matches else 'auto'
    else:
        app_requirements = requirements
    kwargs = {'requirements': app_requirements, 'runtime': runtime, 'prerender': prerender, 'manifest': manifest, 'panel_version': panel_version, 'http_patch': http_patch, 'inline': inline, 'verbose': verbose, 'compiled': compiled}
    if state._is_pyodide:
        files = dict((convert_app(app, dest_path, **kwargs) for app in apps))
    else:
        files = _convert_process_pool(apps, dest_path, max_workers=max_workers, **kwargs)
    if build_index and len(files) >= 1:
        index = make_index(files, manifest=build_pwa, title=title)
        with open(dest_path / 'index.html', 'w') as f:
            f.write(index)
        if verbose:
            print('Successfully wrote index.html.')
    if not build_pwa:
        return
    imgs_path = dest_path / 'images'
    imgs_path.mkdir(exist_ok=True)
    img_rel = []
    for img in PWA_IMAGES:
        with open(imgs_path / img.name, 'wb') as f:
            f.write(img.read_bytes())
        img_rel.append(f'images/{img.name}')
    if verbose:
        print('Successfully wrote icons and images.')
    manifest = build_pwa_manifest(files, title=title, **pwa_config)
    with open(dest_path / 'site.webmanifest', 'w', encoding='utf-8') as f:
        f.write(manifest)
    if verbose:
        print('Successfully wrote site.manifest.')
    worker = SERVICE_WORKER_TEMPLATE.render(uuid=uuid.uuid4().hex, name=title or 'Panel Pyodide App', pre_cache=', '.join([repr(p) for p in img_rel]))
    with open(dest_path / 'serviceWorker.js', 'w', encoding='utf-8') as f:
        f.write(worker)
    if verbose:
        print('Successfully wrote serviceWorker.js.')