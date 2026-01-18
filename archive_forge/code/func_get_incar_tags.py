from __future__ import annotations
import re
import requests
from bs4 import BeautifulSoup
@classmethod
def get_incar_tags(cls):
    """Returns: All incar tags."""
    tags = []
    for page in ['https://www.vasp.at/wiki/index.php/Category:INCAR', 'https://www.vasp.at/wiki/index.php?title=Category:INCAR&pagefrom=ML+FF+LCONF+DISCARD#mw-pages']:
        response = requests.get(page, verify=False)
        soup = BeautifulSoup(response.text)
        for div in soup.findAll('div', {'class': 'mw-category-group'}):
            children = div.findChildren('li')
            for child in children:
                tags.append(child.text.strip())
    return tags