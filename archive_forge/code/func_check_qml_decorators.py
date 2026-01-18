import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
def check_qml_decorators(py_file: Path) -> Tuple[bool, QmlProjectData]:
    """Check if a Python file has QML-decorated classes by running a moc check
    and return whether a class was found and the QML data."""
    data = None
    try:
        cmd = [MOD_CMD, '--quiet', os.fspath(py_file)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            data = json.load(proc.stdout)
            proc.wait()
    except Exception as e:
        t = type(e).__name__
        print(f'{t}: running {MOD_CMD} on {py_file}: {e}', file=sys.stderr)
        sys.exit(1)
    qml_project_data = QmlProjectData()
    if not data:
        return (False, qml_project_data)
    first = data[0]
    class_list = first['classes']
    has_class = _has_qml_decorated_class(class_list)
    if has_class:
        v = first.get(QML_IMPORT_NAME)
        if v:
            qml_project_data.import_name = v
        v = first.get(QML_IMPORT_MAJOR_VERSION)
        if v:
            qml_project_data.import_major_version = v
            qml_project_data.import_minor_version = first.get(QML_IMPORT_MINOR_VERSION)
        v = first.get(QT_MODULES)
        if v:
            qml_project_data.qt_modules = v
    return (has_class, qml_project_data)