import json
from pathlib import Path
import traitlets
from tornado import ioloop
from .handlers import add_handlers
from .manager import LanguageServerManager
from .paths import normalized_uri
def load_jupyter_server_extension(nbapp):
    """create a LanguageServerManager and add handlers"""
    nbapp.add_traits(language_server_manager=traitlets.Instance(LanguageServerManager))
    manager = nbapp.language_server_manager = LanguageServerManager(parent=nbapp)
    contents = nbapp.contents_manager
    page_config = nbapp.web_app.settings.setdefault('page_config_data', {})
    root_uri = ''
    virtual_documents_uri = ''
    if hasattr(contents, 'root_dir'):
        root_uri = normalized_uri(contents.root_dir)
        nbapp.log.debug('[lsp] rootUri will be %s', root_uri)
        virtual_documents_uri = normalized_uri(Path(contents.root_dir) / manager.virtual_documents_dir)
        nbapp.log.debug('[lsp] virtualDocumentsUri will be %s', virtual_documents_uri)
    else:
        nbapp.log.warn('[lsp] %s did not appear to have a root_dir, could not set rootUri', contents)
    page_config.update(rootUri=root_uri, virtualDocumentsUri=virtual_documents_uri)
    add_handlers(nbapp)
    if hasattr(nbapp, 'io_loop'):
        io_loop = nbapp.io_loop
    else:
        io_loop = ioloop.IOLoop.current()
    io_loop.call_later(0, initialize, nbapp, virtual_documents_uri)