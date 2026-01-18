import os
from os.path import dirname, join, exists, abspath
from kivy.clock import Clock
from kivy.compat import PY2
from kivy.properties import ObjectProperty, NumericProperty, \
from kivy.lang import Builder
from kivy.utils import get_hex_from_color, get_color_from_hex
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import AsyncImage, Image
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.anchorlayout import AnchorLayout
from kivy.animation import Animation
from kivy.logger import Logger
from docutils.parsers import rst
from docutils.parsers.rst import roles
from docutils import nodes, frontend, utils
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.roles import set_classes
def dispatch_departure(self, node):
    cls = node.__class__
    if cls is nodes.document:
        self.pop()
    elif cls is nodes.section:
        self.section -= 1
    elif cls is nodes.title:
        assert isinstance(self.current, RstTitle)
        if not self.title:
            self.title = self.text
        self.set_text(self.current, 'title')
        self.pop()
    elif cls is nodes.Text:
        pass
    elif cls is nodes.paragraph:
        self.do_strip_text = False
        assert isinstance(self.current, RstParagraph)
        self.set_text(self.current, 'paragraph')
        self.pop()
    elif cls is nodes.literal_block:
        assert isinstance(self.current, RstLiteralBlock)
        self.set_text(self.current.content, 'literal_block')
        self.pop()
    elif cls is nodes.emphasis:
        self.text += '[/i]'
    elif cls is nodes.strong:
        self.text += '[/b]'
    elif cls is nodes.literal:
        self.text += '[/font]'
    elif cls is nodes.block_quote:
        self.pop()
    elif cls is nodes.enumerated_list:
        self.idx_list = None
        self.pop()
    elif cls is nodes.bullet_list:
        self.pop()
    elif cls is nodes.list_item:
        self.pop()
    elif cls is nodes.system_message:
        self.pop()
    elif cls is nodes.warning:
        self.pop()
    elif cls is nodes.note:
        self.pop()
    elif cls is nodes.definition_list:
        self.pop()
    elif cls is nodes.term:
        assert isinstance(self.current, RstTerm)
        self.set_text(self.current, 'term')
        self.pop()
    elif cls is nodes.definition:
        self.pop()
    elif cls is nodes.field_list:
        self.pop()
    elif cls is nodes.field_name:
        assert isinstance(self.current, RstFieldName)
        self.set_text(self.current, 'field_name')
        self.pop()
    elif cls is nodes.field_body:
        self.pop()
    elif cls is nodes.table:
        self.pop()
    elif cls is nodes.colspec:
        pass
    elif cls is nodes.entry:
        self.pop()
    elif cls is nodes.reference:
        self.text += '[/color][/ref]'
    elif cls is nodes.footnote:
        self.pop()
        self.set_text(self.current, 'link')
    elif cls is nodes.footnote_reference:
        self.text += '[/color][/ref]'
        self.text += '&br;'
    elif cls is role_doc:
        docname = self.text[self.doc_index:]
        rst_docname = docname
        if rst_docname.endswith('.rst'):
            docname = docname[:-4]
        else:
            rst_docname += '.rst'
        filename = self.root.resolve_path(rst_docname)
        self.root.preload(filename)
        title = docname
        if filename in self.root.toctrees:
            toctree = self.root.toctrees[filename]
            if len(toctree):
                title = toctree[0]['title']
        text = '[ref=%s]%s[/ref]' % (rst_docname, self.colorize(title, 'link'))
        self.text = self.text[:self.doc_index] + text
    elif cls is role_video:
        width = node['width'] if 'width' in node.attlist() else 400
        height = node['height'] if 'height' in node.attlist() else 300
        uri = node['source']
        if uri.startswith('/') and self.root.document_root:
            uri = join(self.root.document_root, uri[1:])
        video = RstVideoPlayer(source=uri, size_hint=(None, None), size=(width, height))
        anchor = AnchorLayout(size_hint_y=None, height=height + 20)
        anchor.add_widget(video)
        self.current.add_widget(anchor)