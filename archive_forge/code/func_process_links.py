from __future__ import absolute_import, division, print_function
import base64
import re
import textwrap
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import unquote
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import ModuleFailException
def process_links(info, callback):
    """
    Process link header, calls callback for every link header with the URL and relation as options.
    """
    if 'link' in info:
        link = info['link']
        for url, relation in re.findall('<([^>]+)>;\\s*rel="(\\w+)"', link):
            callback(unquote(url), relation)