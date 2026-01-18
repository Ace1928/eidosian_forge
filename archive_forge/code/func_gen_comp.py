import asyncio
import os
import sys
from IPython.core.debugger import Pdb
from IPython.core.completer import IPCompleter
from .ptutils import IPythonPTCompleter
from .shortcuts import create_ipython_shortcuts
from . import embed
from pathlib import Path
from pygments.token import Token
from prompt_toolkit.application import create_app_session
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import InMemoryHistory, FileHistory
from concurrent.futures import ThreadPoolExecutor
from prompt_toolkit import __version__ as ptk_version
def gen_comp(self, text):
    return [m for m in methods_names if m.startswith(text)]