from __future__ import annotations
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
def make_output(graph, raw, filename):
    """Produce output for pass_manager."""
    if raw:
        if filename:
            graph.write(filename, format='raw')
            return None
        else:
            raise VisualizationError('if format=raw, then a filename is required.')
    if not _optionals.HAS_PIL and filename:
        graph.write_png(filename)
        return None
    _optionals.HAS_PIL.require_now('pass manager drawer')
    with tempfile.TemporaryDirectory() as tmpdirname:
        from PIL import Image
        tmppath = os.path.join(tmpdirname, 'pass_manager.png')
        graph.write_png(tmppath)
        image = Image.open(tmppath)
        os.remove(tmppath)
        if filename:
            image.save(filename, 'PNG')
        return image