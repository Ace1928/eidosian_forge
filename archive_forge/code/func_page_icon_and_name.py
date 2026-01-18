from __future__ import annotations
import re
import threading
from pathlib import Path
from typing import Any, Callable, Final, cast
from blinker import Signal
from streamlit.logger import get_logger
from streamlit.string_util import extract_leading_emoji
from streamlit.util import calc_md5
def page_icon_and_name(script_path: Path) -> tuple[str, str]:
    """Compute the icon and name of a page from its script path.

    This is *almost* the page name displayed in the nav UI, but it has
    underscores instead of spaces. The reason we do this is because having
    spaces in URLs both looks bad and is hard to deal with due to the need to
    URL-encode them. To solve this, we only swap the underscores for spaces
    right before we render page names.
    """
    extraction = re.search(PAGE_FILENAME_REGEX, script_path.name)
    if extraction is None:
        return ('', '')
    extraction: re.Match[str] = cast(Any, extraction)
    icon_and_name = re.sub('[_ ]+', '_', extraction.group(2)).strip() or extraction.group(1)
    return extract_leading_emoji(icon_and_name)