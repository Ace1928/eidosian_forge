import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def set_test_template(self, template):
    self._test_template = self._is_template_set(template)