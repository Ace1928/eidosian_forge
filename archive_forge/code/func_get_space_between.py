import re
import lxml
import lxml.etree
from lxml.html.clean import Cleaner
def get_space_between(text, prev):
    if not text:
        return ' '
    return ' ' if should_add_space(text, prev) else ''