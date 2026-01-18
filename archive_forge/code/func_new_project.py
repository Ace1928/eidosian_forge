import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple
from PySide6.QtWidgets import QApplication, QMainWindow
import QtQuick.Controls
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine
def new_project(directory_s: str, project_type: ProjectType=ProjectType.WIDGET_FORM) -> int:
    directory = Path(directory_s)
    if directory.exists():
        print(f'{directory_s} already exists.', file=sys.stderr)
        return -1
    directory.mkdir(parents=True)
    if project_type == ProjectType.WIDGET_FORM:
        project = _ui_form_project()
    elif project_type == ProjectType.QUICK:
        project = _qml_project()
    else:
        project = _widget_project()
    _write_project(directory, project)
    if project_type == ProjectType.WIDGET_FORM:
        print(f'Run "pyside6-project build {directory_s}" to build the project')
    print(f'Run "python {directory.name}{os.sep}main.py" to run the project')
    return 0