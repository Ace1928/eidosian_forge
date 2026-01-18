from __future__ import annotations
import errno
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any
from jupyter_client.kernelspec import KernelSpecManager
from traitlets import Unicode
from traitlets.config import Application
def get_kernel_dict(extra_arguments: list[str] | None=None, python_arguments: list[str] | None=None) -> dict[str, Any]:
    """Construct dict for kernel.json"""
    return {'argv': make_ipkernel_cmd(extra_arguments=extra_arguments, python_arguments=python_arguments), 'display_name': 'Python %i (ipykernel)' % sys.version_info[0], 'language': 'python', 'metadata': {'debugger': True}}