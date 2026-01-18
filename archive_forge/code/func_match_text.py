import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_text(self):
    match = self.match("\n                (.*?)         # anything, followed by:\n                (\n                 (?<=\\n)(?=[ \\t]*(?=%|\\#\\#)) # an eval or line-based\n                                             # comment preceded by a\n                                             # consumed newline and whitespace\n                 |\n                 (?=\\${)      # an expression\n                 |\n                 (?=</?[%&])  # a substitution or block or call start or end\n                              # - don't consume\n                 |\n                 (\\\\\\r?\\n)    # an escaped newline  - throw away\n                 |\n                 \\Z           # end of string\n                )", re.X | re.S)
    if match:
        text = match.group(1)
        if text:
            self.append_node(parsetree.Text, text)
        return True
    else:
        return False