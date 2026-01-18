import gettext
import io
import sys
import json
import logging
import threading
import tempfile
import os
import shutil
import signal
import socket
import webbrowser
import errno
import random
import jinja2
import tornado.ioloop
import tornado.web
from traitlets.config.application import Application
from traitlets.config.loader import Config
from traitlets import Unicode, Integer, Bool, Dict, List, Callable, default, Type, Bytes
from jupyter_server.services.contents.largefilemanager import LargeFileManager
from jupyter_server.services.kernels.handlers import KernelHandler
from jupyter_server.base.handlers import path_regex, FileFindHandler
from jupyter_server.config_manager import recursive_update
from jupyterlab_server.themes_handler import ThemesHandler
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_core.paths import jupyter_config_path, jupyter_path
from .paths import ROOT, STATIC_ROOT, collect_template_paths, collect_static_paths
from .handler import VoilaHandler
from .treehandler import VoilaTreeHandler
from ._version import __version__
from .static_file_handler import MultiStaticFileHandler, TemplateStaticFileHandler, WhiteListFileHandler
from .configuration import VoilaConfiguration
from .execute import VoilaExecutor
from .exporter import VoilaExporter
from .shutdown_kernel_handler import VoilaShutdownKernelHandler
from .voila_kernel_manager import voila_kernel_manager_factory
from .request_info_handler import RequestInfoSocketHandler
from .utils import create_include_assets_functions
def setup_template_dirs(self):
    if self.voila_configuration.template:
        template_name = self.voila_configuration.template
        self.template_paths = collect_template_paths(['voila', 'nbconvert'], template_name, prune=True)
        self.static_paths = collect_static_paths(['voila', 'nbconvert'], template_name)
        if JUPYTER_SERVER_2:
            self.static_paths.append(DEFAULT_STATIC_FILES_PATH)
        conf_paths = [os.path.join(d, 'conf.json') for d in self.template_paths]
        for p in conf_paths:
            if os.path.exists(p):
                with open(p) as json_file:
                    conf = json.load(json_file)
                if 'traitlet_configuration' in conf:
                    recursive_update(conf['traitlet_configuration'], self.voila_configuration.config.VoilaConfiguration)
                    self.voila_configuration.config.VoilaConfiguration = Config(conf['traitlet_configuration'])
    self.log.debug('using template: %s', self.voila_configuration.template)
    self.log.debug('template paths:\n\t%s', '\n\t'.join(self.template_paths))
    self.log.debug('static paths:\n\t%s', '\n\t'.join(self.static_paths))
    if self.notebook_path and (not os.path.exists(self.notebook_path)):
        raise ValueError('Notebook not found: %s' % self.notebook_path)