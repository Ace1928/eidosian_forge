import os
import sys
import warnings
from pathlib import Path
from threading import local
from IPython.core import page, payloadpage
from IPython.core.autocall import ZMQExitAutocall
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magics import CodeMagics, MacroToEdit  # type:ignore[attr-defined]
from IPython.core.usage import default_banner
from IPython.display import Javascript, display
from IPython.utils import openpy
from IPython.utils.process import arg_split, system  # type:ignore[attr-defined]
from jupyter_client.session import Session, extract_header
from jupyter_core.paths import jupyter_runtime_dir
from traitlets import Any, CBool, CBytes, Dict, Instance, Type, default, observe
from ipykernel import connect_qtconsole, get_connection_file, get_connection_info
from ipykernel.displayhook import ZMQShellDisplayHook
from ipykernel.jsonutil import encode_images, json_clean
@line_magic
def connect_info(self, arg_s):
    """Print information for connecting other clients to this kernel

        It will print the contents of this session's connection file, as well as
        shortcuts for local clients.

        In the simplest case, when called from the most recently launched kernel,
        secondary clients can be connected, simply with:

        $> jupyter <app> --existing

        """
    try:
        connection_file = get_connection_file()
        info = get_connection_info(unpack=False)
    except Exception as e:
        warnings.warn('Could not get connection info: %r' % e, stacklevel=2)
        return
    if jupyter_runtime_dir() == str(Path(connection_file).parent):
        connection_file = Path(connection_file).name
    assert isinstance(info, str)
    print(info + '\n')
    print(f'Paste the above JSON into a file, and connect with:\n    $> jupyter <app> --existing <file>\nor, if you are local, you can connect with just:\n    $> jupyter <app> --existing {connection_file}\nor even just:\n    $> jupyter <app> --existing\nif this is the most recent Jupyter kernel you have started.')