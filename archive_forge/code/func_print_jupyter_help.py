from __future__ import annotations
import re
import requests
from bs4 import BeautifulSoup
def print_jupyter_help(self, tag):
    """
        Display HTML help in ipython notebook.

        Args:
            tag (str): Tag used in VASP.
        """
    html_str = self.get_help(tag, 'html')
    from IPython.core.display import HTML, display
    display(HTML(html_str))