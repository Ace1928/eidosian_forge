import re
from formencode.rewritingparser import RewritingParser, html_quote
def selected_multiple(self, obj, value):
    """
        Returns true/false if obj indicates that value should be
        selected.  If obj has a __contains__ method it is used, otherwise
        identity is used.
        """
    if obj is None:
        return False
    if isinstance(obj, str):
        return obj == value
    if hasattr(obj, '__contains__'):
        if value in obj:
            return True
    if hasattr(obj, '__iter__'):
        for inner in obj:
            if self.str_compare(inner, value):
                return True
    return self.str_compare(obj, value)