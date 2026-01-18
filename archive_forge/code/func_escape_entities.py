import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
def escape_entities(self, text):
    """Replace characters with their character entity references.

        Only characters corresponding to a named entity are replaced.
        """
    return str(text).translate(self.codepoint2entity)