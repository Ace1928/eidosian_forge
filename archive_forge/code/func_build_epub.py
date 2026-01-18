import html
import os
import re
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging, status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import copyfile, ensuredir
def build_epub(self) -> None:
    """Write the epub file.

        It is a zip file with the mimetype file stored uncompressed as the first
        entry.
        """
    outname = self.config.epub_basename + '.epub'
    logger.info(__('writing %s file...'), outname)
    epub_filename = path.join(self.outdir, outname)
    with ZipFile(epub_filename, 'w', ZIP_DEFLATED) as epub:
        epub.write(path.join(self.outdir, 'mimetype'), 'mimetype', ZIP_STORED)
        for filename in ('META-INF/container.xml', 'content.opf', 'toc.ncx'):
            epub.write(path.join(self.outdir, filename), filename, ZIP_DEFLATED)
        for filename in self.files:
            epub.write(path.join(self.outdir, filename), filename, ZIP_DEFLATED)