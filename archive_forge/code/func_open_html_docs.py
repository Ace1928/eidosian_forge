import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
def open_html_docs(page):
    doc_dir = os.path.join(os.path.dirname(snappy_dir), 'doc')
    if sys.platform.startswith('linux'):
        safe_doc_dir = os.path.join(os.environ['HOME'], 'Downloads', f'SnapPy_{version}_help_f8205a5')
        if os.path.exists(safe_doc_dir):
            shutil.rmtree(safe_doc_dir)
        shutil.copytree(doc_dir, safe_doc_dir)
    else:
        safe_doc_dir = doc_dir
    path = os.path.join(safe_doc_dir, page)
    if os.path.exists(path):
        url = 'file:' + pathname2url(path)
        try:
            webbrowser.open_new_tab(url)
        except webbrowser.Error:
            from tkinter import messagebox
            messagebox.showwarning('Error', 'Failed to open the documentation file.')
    else:
        from tkinter import messagebox
        messagebox.showwarning('Not found!', 'The file %s does not exist.' % path)