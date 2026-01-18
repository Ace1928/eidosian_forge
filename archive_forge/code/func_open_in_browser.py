import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def open_in_browser(doc, encoding=None):
    """
    Open the HTML document in a web browser, saving it to a temporary
    file to open it.  Note that this does not delete the file after
    use.  This is mainly meant for debugging.
    """
    import os
    import webbrowser
    import tempfile
    if not isinstance(doc, etree._ElementTree):
        doc = etree.ElementTree(doc)
    handle, fn = tempfile.mkstemp(suffix='.html')
    f = os.fdopen(handle, 'wb')
    try:
        doc.write(f, method='html', encoding=encoding or doc.docinfo.encoding or 'UTF-8')
    finally:
        f.close()
    url = 'file://' + fn.replace(os.path.sep, '/')
    print(url)
    webbrowser.open(url)