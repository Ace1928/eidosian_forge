import html
import html.entities
import re
from urllib.parse import quote, unquote
def strip_html(s):
    s = re.sub('<.*?>', '', s)
    s = html_unquote(s)
    return s