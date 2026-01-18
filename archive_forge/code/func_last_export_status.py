import os
import sys
import time
import traceback
import param
from IPython.display import Javascript, display
from nbconvert import HTMLExporter, NotebookExporter
from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbformat import reader
from ..core.io import FileArchive, Pickler
from ..plotting.renderer import HTML_TAGS, MIME_TYPES
from .preprocessors import Substitute
def last_export_status(self):
    """Helper to show the status of the last call to the export method."""
    if self.export_success is True:
        print('The last call to holoviews.archive.export was successful.')
        return
    elif self.export_success is None:
        print('Status of the last call to holoviews.archive.export is unknown.\n(Re-execute this method once kernel status is idle.)')
        return
    print('The last call to holoviews.archive.export was unsuccessful.')
    if self.traceback is None:
        print('\n<No traceback captured>')
    else:
        print('\n' + self.traceback)