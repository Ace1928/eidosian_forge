import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
@classmethod
def enable_offline(cls, offline: bool=True):
    """
        Configure JupyterChart's offline behavior

        Parameters
        ----------
        offline: bool
            If True, configure JupyterChart to operate in offline mode where JavaScript
            dependencies are loaded from vl-convert.
            If False, configure it to operate in online mode where JavaScript dependencies
            are loaded from CDN dynamically. This is the default behavior.
        """
    from altair.utils._importers import import_vl_convert, vl_version_for_vl_convert
    if offline:
        if cls._is_offline:
            return
        vlc = import_vl_convert()
        src_lines = load_js_src().split('\n')
        while src_lines and (len(src_lines[0].strip()) == 0 or src_lines[0].startswith('import') or src_lines[0].startswith('//')):
            src_lines.pop(0)
        src = '\n'.join(src_lines)
        bundled_src = vlc.javascript_bundle(src, vl_version=vl_version_for_vl_convert())
        cls._esm = bundled_src
        cls._is_offline = True
    else:
        cls._esm = load_js_src()
        cls._is_offline = False