import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
def init_tokenizer(self) -> None:
    if not janome_module:
        raise RuntimeError('Janome is not available')
    self.tokenizer = janome.tokenizer.Tokenizer(udic=self.user_dict, udic_enc=self.user_dict_enc)