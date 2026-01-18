import re
import docutils
from docutils import nodes, writers, languages
def deunicode(self, text):
    text = text.replace('\xa0', '\\ ')
    text = text.replace('†', '\\(dg')
    return text