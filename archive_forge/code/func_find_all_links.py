import logging
import re
from typing import List, Optional, Sequence, Union
from urllib.parse import urljoin, urlparse
def find_all_links(raw_html: str, *, pattern: Union[str, re.Pattern, None]=None) -> List[str]:
    """Extract all links from a raw html string.

    Args:
        raw_html: original html.
        pattern: Regex to use for extracting links from raw html.

    Returns:
        List[str]: all links
    """
    pattern = pattern or DEFAULT_LINK_REGEX
    return list(set(re.findall(pattern, raw_html)))