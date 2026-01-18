import html
from os import path
from typing import Any, Dict, List, NamedTuple, Set, Tuple
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import _epub_base
from sphinx.config import ENUM, Config
from sphinx.locale import __
from sphinx.util import logging, xmlname_checker
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import make_filename
def content_metadata(self) -> Dict[str, Any]:
    """Create a dictionary with all metadata for the content.opf
        file properly escaped.
        """
    writing_mode = self.config.epub_writing_mode
    metadata = super().content_metadata()
    metadata['description'] = html.escape(self.config.epub_description)
    metadata['contributor'] = html.escape(self.config.epub_contributor)
    metadata['page_progression_direction'] = PAGE_PROGRESSION_DIRECTIONS.get(writing_mode)
    metadata['ibook_scroll_axis'] = IBOOK_SCROLL_AXIS.get(writing_mode)
    metadata['date'] = html.escape(format_date('%Y-%m-%dT%H:%M:%SZ', language='en'))
    metadata['version'] = html.escape(self.config.version)
    metadata['epub_version'] = self.config.epub_version
    return metadata