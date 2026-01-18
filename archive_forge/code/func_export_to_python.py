import argparse
import os
import sys
from argparse import RawTextHelpFormatter
from . import examples
def export_to_python(filename=None, preprocessors=None):
    if preprocessors is None:
        preprocessors = _PREPROCESSORS.copy()
    filename = filename if filename else sys.argv[1]
    with open(filename) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        exporter = nbconvert.PythonExporter()
        for preprocessor in preprocessors:
            exporter.register_preprocessor(preprocessor)
        source, meta = exporter.from_notebook_node(nb)
        return source