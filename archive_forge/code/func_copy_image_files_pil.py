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
def copy_image_files_pil(self) -> None:
    """Copy images using Pillow, the Python Imaging Library.
        The method tries to read and write the files with Pillow, converting
        the format and resizing the image if necessary/possible.
        """
    ensuredir(path.join(self.outdir, self.imagedir))
    for src in status_iterator(self.images, __('copying images... '), 'brown', len(self.images), self.app.verbosity):
        dest = self.images[src]
        try:
            img = Image.open(path.join(self.srcdir, src))
        except OSError:
            if not self.is_vector_graphics(src):
                logger.warning(__('cannot read image file %r: copying it instead'), path.join(self.srcdir, src))
            try:
                copyfile(path.join(self.srcdir, src), path.join(self.outdir, self.imagedir, dest))
            except OSError as err:
                logger.warning(__('cannot copy image file %r: %s'), path.join(self.srcdir, src), err)
            continue
        if self.config.epub_fix_images:
            if img.mode in ('P',):
                img = img.convert()
        if self.config.epub_max_image_width > 0:
            width, height = img.size
            nw = self.config.epub_max_image_width
            if width > nw:
                nh = height * nw / width
                img = img.resize((nw, nh), Image.BICUBIC)
        try:
            img.save(path.join(self.outdir, self.imagedir, dest))
        except OSError as err:
            logger.warning(__('cannot write image file %r: %s'), path.join(self.srcdir, src), err)