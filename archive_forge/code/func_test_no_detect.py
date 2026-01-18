from jupyter_lsp.specs.r_languageserver import RLanguageServer
from jupyter_lsp.specs.utils import PythonModuleSpec
def test_no_detect(manager):
    """should not enable anything by default"""
    manager.autodetect = False
    manager.initialize()
    assert not manager.language_servers
    assert not manager.sessions