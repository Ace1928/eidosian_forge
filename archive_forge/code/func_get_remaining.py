import io
import sys
from typing import Any, List, Optional, Tuple
import dns.exception
import dns.name
import dns.ttl
def get_remaining(self, max_tokens: Optional[int]=None) -> List[Token]:
    """Return the remaining tokens on the line, until an EOL or EOF is seen.

        max_tokens: If not None, stop after this number of tokens.

        Returns a list of tokens.
        """
    tokens = []
    while True:
        token = self.get()
        if token.is_eol_or_eof():
            self.unget(token)
            break
        tokens.append(token)
        if len(tokens) == max_tokens:
            break
    return tokens