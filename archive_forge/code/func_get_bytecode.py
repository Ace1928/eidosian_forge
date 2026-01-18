from __future__ import annotations
import os.path
import threading
from typing import Any
from streamlit import config
from streamlit.runtime.scriptrunner import magic
from streamlit.source_util import open_python_file
def get_bytecode(self, script_path: str) -> Any:
    """Return the bytecode for the Python script at the given path.

        If the bytecode is not already in the cache, the script will be
        compiled first.

        Raises
        ------
        Any Exception raised while reading or compiling the script.

        Notes
        -----
        Threading: SAFE. May be called on any thread.
        """
    script_path = os.path.abspath(script_path)
    with self._lock:
        bytecode = self._cache.get(script_path, None)
        if bytecode is not None:
            return bytecode
        with open_python_file(script_path) as f:
            filebody = f.read()
        if config.get_option('runner.magicEnabled'):
            filebody = magic.add_magic(filebody, script_path)
        bytecode = compile(filebody, script_path, mode='exec', flags=0, dont_inherit=1, optimize=-1)
        self._cache[script_path] = bytecode
        return bytecode