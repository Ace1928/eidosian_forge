from typing import Union, Optional, Mapping, Dict, Tuple, Iterator
from lark import Tree, Transformer
from lark.exceptions import MissingVariableError
def apply_vars(self, vars: Mapping[str, Tree[str]]) -> Tree[str]:
    """Apply vars to the template tree
        """
    return _ReplaceVars(self.conf, vars).transform(self.tree)