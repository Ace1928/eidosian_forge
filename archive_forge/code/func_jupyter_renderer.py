import os
from typing import Dict
from ...utils.mimebundle import spec_to_mimebundle
from ..display import (
from .schema import SCHEMA_VERSION
from typing import Final
def jupyter_renderer(spec: dict, **metadata):
    """Render chart using the JupyterChart Jupyter Widget"""
    from altair import Chart, JupyterChart
    offline = metadata.get('offline', False)
    JupyterChart.enable_offline(offline=offline)
    embed_options = metadata.get('embed_options', None)
    return JupyterChart(chart=Chart.from_dict(spec), embed_options=embed_options)._repr_mimebundle_()