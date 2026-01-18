import os
import re
from pathlib import Path
from typing import Union
from urllib.parse import unquote, urlparse
def normalized_uri(root_dir):
    """Attempt to make an LSP rootUri from a ContentsManager root_dir

    Special care must be taken around windows paths: the canonical form of
    windows drives and UNC paths is lower case
    """
    root_uri = Path(root_dir).expanduser().resolve().as_uri()
    root_uri = re.sub(RE_PATH_ANCHOR, lambda m: 'file://{}'.format(m.group(1).lower()), root_uri)
    return root_uri