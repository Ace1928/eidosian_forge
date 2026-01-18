import re
from formencode.rewritingparser import RewritingParser, html_quote
def str_compare(self, str1, str2):
    """
        Compare the two objects as strings (coercing to strings if necessary).
        Also uses encoding to compare the strings.
        """
    if not isinstance(str1, str):
        str1 = str(str1)
    if type(str1) is type(str2):
        return str1 == str2
    if isinstance(str1, str):
        str1 = str1.encode(self.encoding or self.default_encoding)
    else:
        str2 = str2.encode(self.encoding or self.default_encoding)
    return str1 == str2