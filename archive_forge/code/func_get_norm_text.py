from typing import Iterable, List, Optional, Tuple
import pkg_resources
from antlr4 import InputStream, Token
from antlr4.Token import CommonToken
from antlr4.tree.Tree import ParseTree, TerminalNode
from packaging import version
from fugue_sql_antlr._parser.fugue_sqlParser import fugue_sqlParser
from fugue_sql_antlr._parser.sa_fugue_sql import (
from fugue_sql_antlr.constants import (
def get_norm_text(self, node: Optional[ParseTree], delimiter: str=' ', upper_keyword: bool=False) -> str:

    def _get_token_str(token: Token):
        s = self.code[token.start:token.stop + 1]
        if upper_keyword and self._ignore_case and hasattr(token, 'is_keyword'):
            return s.upper()
        return s
    if node is None:
        return ''
    tp = self._get_range(node)
    if tp is None:
        return ''
    if tp[0] is tp[1]:
        return self.code[tp[0].start:tp[0].stop + 1]
    return delimiter.join((_get_token_str(self._tokens[i]) for i in range(tp[0]._arr_pos, tp[1]._arr_pos + 1)))