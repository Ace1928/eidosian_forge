import io
import sys
from typing import Any, List, Optional, Tuple
import dns.exception
import dns.name
import dns.ttl
def get_eol_as_token(self) -> Token:
    """Read the next token and raise an exception if it isn't EOL or
        EOF.

        Returns a string.
        """
    token = self.get()
    if not token.is_eol_or_eof():
        raise dns.exception.SyntaxError('expected EOL or EOF, got %d "%s"' % (token.ttype, token.value))
    return token