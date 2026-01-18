from __future__ import annotations
import html
import os
from os import path
from typing import Any
from docutils import nodes
from docutils.nodes import Element, Node, document
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename_from_project, relpath
from sphinx.util.template import SphinxRenderer
def build_hhx(self, outdir: str | os.PathLike[str], outname: str) -> None:
    logger.info(__('writing index file...'))
    index = IndexEntries(self.env).create_index(self)
    filename = path.join(outdir, outname + '.hhk')
    with open(filename, 'w', encoding=self.encoding, errors='xmlcharrefreplace') as f:
        f.write('<UL>\n')

        def write_index(title: str, refs: list[tuple[str, str]], subitems: list[tuple[str, list[tuple[str, str]]]]) -> None:

            def write_param(name: str, value: str) -> None:
                item = '    <param name="%s" value="%s">\n' % (name, value)
                f.write(item)
            title = chm_htmlescape(title, True)
            f.write('<LI> <OBJECT type="text/sitemap">\n')
            write_param('Keyword', title)
            if len(refs) == 0:
                write_param('See Also', title)
            elif len(refs) == 1:
                write_param('Local', refs[0][1])
            else:
                for i, ref in enumerate(refs):
                    write_param('Name', '[%d] %s' % (i, ref[1]))
                    write_param('Local', ref[1])
            f.write('</OBJECT>\n')
            if subitems:
                f.write('<UL> ')
                for subitem in subitems:
                    write_index(subitem[0], subitem[1], [])
                f.write('</UL>')
        for key, group in index:
            for title, (refs, subitems, key_) in group:
                write_index(title, refs, subitems)
        f.write('</UL>\n')