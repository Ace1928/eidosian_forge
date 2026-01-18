import os
import sys
from nbformat import NotebookNode
from traitlets.config import get_config
from traitlets.log import get_logger
from traitlets.utils.importstring import import_item
from .exporter import Exporter
def get_exporter(name, config=get_config()):
    """Given an exporter name or import path, return a class ready to be instantiated

    Raises ExporterName if exporter is not found or ExporterDisabledError if not enabled
    """
    if name == 'ipynb':
        name = 'notebook'
    try:
        exporters = entry_points(group='nbconvert.exporters')
        items = [e for e in exporters if e.name == name or e.name == name.lower()]
        exporter = items[0].load()
        if getattr(exporter(config=config), 'enabled', True):
            return exporter
        raise ExporterDisabledError('Exporter "%s" disabled in configuration' % name)
    except IndexError:
        pass
    if '.' in name:
        try:
            exporter = import_item(name)
            if getattr(exporter(config=config), 'enabled', True):
                return exporter
            raise ExporterDisabledError('Exporter "%s" disabled in configuration' % name)
        except ImportError:
            log = get_logger()
            log.error('Error importing %s', name, exc_info=True)
    msg = 'Unknown exporter "{}", did you mean one of: {}?'.format(name, ', '.join(get_export_names()))
    raise ExporterNameError(msg)