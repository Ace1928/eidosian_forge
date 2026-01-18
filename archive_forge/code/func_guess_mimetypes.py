import os
import re
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import epoch_to_rfc1123, logging, requests, rfc1123_to_epoch, sha1
from sphinx.util.images import get_image_extension, guess_mimetype, parse_data_uri
from sphinx.util.osutil import ensuredir
def guess_mimetypes(self, node: nodes.image) -> List[str]:
    if '?' in node['candidates']:
        return []
    elif '*' in node['candidates']:
        return [guess_mimetype(node['uri'])]
    else:
        return node['candidates'].keys()