import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any, Optional
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes
def launch_file_browser(app: QApplication) -> Optional[Path]:
    """
    Launches the file browser GUI, returning the selected directory path.

    Args:
        app (QApplication): The Qt application instance.

    Returns:
        Optional[Path]: The selected directory path from the file browser GUI, or None if no selection was made.
    """
    try:
        file_browser = FileBrowserMainWindow()
        file_browser.show()
        app.exec_()
        selected_indexes = file_browser.tree_view.selectedIndexes()
        if selected_indexes:
            selected_path = Path(file_browser.model.filePath(selected_indexes[0]))
            return selected_path
        return None
    except Exception as e:
        logger.error(f'Error launching file browser: {e}')
        return None