import re
import sys
def html_doctype_matches(text):
    """Check if the file looks like it has a html doctype."""
    return doctype_matches(text, 'html')