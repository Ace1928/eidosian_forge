import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def qt_metatype_json_dir() -> Path:
    """Return the location of the Qt QML metatype files."""
    global _qt_metatype_json_dir
    if not _qt_metatype_json_dir:
        qt_dir = package_dir()
        if sys.platform != 'win32':
            qt_dir /= 'Qt'
        metatypes_dir = qt_dir / 'metatypes'
        if metatypes_dir.is_dir():
            _qt_metatype_json_dir = metatypes_dir
        else:
            print(f'Falling back to {QTPATHS_CMD} to determine metatypes directory.', file=sys.stderr)
            _qt_metatype_json_dir = Path(qtpaths()['QT_INSTALL_ARCHDATA']) / 'metatypes'
    return _qt_metatype_json_dir