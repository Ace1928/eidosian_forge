import builtins as builtin_mod
import sys
import types
from pathlib import Path
from . import tools
from IPython.core import page
from IPython.utils import io
from IPython.terminal.interactiveshell import TerminalInteractiveShell
def start_ipython():
    """Start a global IPython shell, which we need for IPython-specific syntax.
    """
    global get_ipython
    if hasattr(start_ipython, 'already_called'):
        return
    start_ipython.already_called = True
    _displayhook = sys.displayhook
    _excepthook = sys.excepthook
    _main = sys.modules.get('__main__')
    config = tools.default_config()
    config.TerminalInteractiveShell.simple_prompt = True
    shell = TerminalInteractiveShell.instance(config=config)
    shell.tempfiles.append(Path(config.HistoryManager.hist_file))
    shell.builtin_trap.activate()
    shell.system = types.MethodType(xsys, shell)
    shell._showtraceback = types.MethodType(_showtraceback, shell)
    sys.modules['__main__'] = _main
    sys.displayhook = _displayhook
    sys.excepthook = _excepthook
    _ip = shell
    get_ipython = _ip.get_ipython
    builtin_mod._ip = _ip
    builtin_mod.ip = _ip
    builtin_mod.get_ipython = get_ipython

    def nopage(strng, start=0, screen_lines=0, pager_cmd=None):
        if isinstance(strng, dict):
            strng = strng.get('text/plain', '')
        print(strng)
    page.orig_page = page.pager_page
    page.pager_page = nopage
    return _ip