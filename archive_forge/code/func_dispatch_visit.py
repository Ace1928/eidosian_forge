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
def dispatch_visit(self, node):
    cls = node.__class__
    if cls is nodes.document:
        self.push(self.root.content)
        self.brute_refs(node)
    elif cls is nodes.comment:
        return
    elif cls is nodes.section:
        self.section += 1
    elif cls is nodes.substitution_definition:
        name = node.attributes['names'][0]
        self.substitution[name] = node.children[0]
    elif cls is nodes.substitution_reference:
        node = self.substitution[node.attributes['refname']]
        if isinstance(node, nodes.Text):
            self.text += node
    elif cls is nodes.footnote:
        text = ''
        foot = RstFootnote()
        ids = node.attributes['ids']
        self.current.add_widget(foot)
        self.push(foot)
        auto = ''
        if 'auto' in node.attributes:
            auto = node.attributes['auto']
        if auto == 1:
            self.footnotes['autonum'] += 1
            name = str(self.footnotes['autonum'])
            node_id = node.attributes['ids'][0]
        elif auto == '*':
            autosym = self.footnotes['autosym']
            name = self.footlist[autosym % 10] * (int(autosym / 10) + 1)
            self.footnotes['autosym'] += 1
            node_id = node.attributes['ids'][0]
        else:
            name = node.attributes['names'][0]
            node_id = node['ids'][0]
        link = self.root.refs_assoc.get(name, '')
        ref = self.root.refs_assoc.get('backref' + name, '')
        colorized = self.colorize(name, 'link') if ref else name
        if not ref:
            text = '&bl;%s&br;' % colorized
        elif ref and isinstance(ref, list):
            ref_block = ['[ref=%s][u]%s[/u][/ref]' % (r, i + 1) for i, r in enumerate(ref)]
            self.foot_refblock = ''.join(['[i]( ', ', '.join(ref_block), ' )[/i]'])
            text = '[anchor=%s]&bl;%s&br;' % (node['ids'][0], colorized)
        else:
            text = '[anchor=%s][ref=%s]&bl;%s&br;[/ref]' % (node['ids'][0], ref, colorized)
        name = RstFootName(document=self.root, text=text)
        self.current.add_widget(name)
        self.root.add_anchors(name)
        name.bind(on_ref_press=self.root.on_ref_press)
    elif cls is nodes.footnote_reference:
        self.text += '&bl;'
        text = ''
        name = ''
        auto = ''
        if 'auto' in node.attributes:
            auto = node.attributes['auto']
        if auto == 1:
            self.footnotes['autonum_ref'] += 1
            name = str(self.footnotes['autonum_ref'])
            node_id = node.attributes['ids'][0]
        elif auto == '*':
            autosym = self.footnotes['autosym_ref']
            name = self.footlist[autosym % 10] * (int(autosym / 10) + 1)
            self.footnotes['autosym_ref'] += 1
            node_id = node.attributes['ids'][0]
        else:
            name = node.children[0]
            node_id = node['ids'][0]
        text += name
        refs = self.root.refs_assoc.get(name, '')
        if not refs and auto in (1, '*'):
            raise Exception('Too many autonumbered or autosymboled footnote references!')
        text = '[anchor=%s][ref=%s][color=%s]%s' % (node_id, refs, self.root.colors.get('link', self.root.colors.get('paragraph')), text)
        self.text += text
        self.text_have_anchor = True
    elif cls is nodes.title:
        label = RstTitle(section=self.section, document=self.root)
        self.current.add_widget(label)
        self.push(label)
    elif cls is nodes.Text:
        if hasattr(node, 'parent'):
            if node.parent.tagname == 'substitution_definition':
                return
            elif node.parent.tagname == 'substitution_reference':
                return
            elif node.parent.tagname == 'comment':
                return
            elif node.parent.tagname == 'footnote_reference':
                return
        if self.do_strip_text:
            node = node.replace('\n', ' ')
            node = node.replace('  ', ' ')
            node = node.replace('\t', ' ')
            node = node.replace('  ', ' ')
            if node.startswith(' '):
                node = ' ' + node.lstrip(' ')
            if node.endswith(' '):
                node = node.rstrip(' ') + ' '
            if self.text.endswith(' ') and node.startswith(' '):
                node = node[1:]
        self.text += node
    elif cls is nodes.paragraph:
        self.do_strip_text = True
        if isinstance(node.parent, nodes.footnote):
            if self.foot_refblock:
                self.text = self.foot_refblock + ' '
            self.foot_refblock = None
        label = RstParagraph(document=self.root)
        if isinstance(self.current, RstEntry):
            label.mx = 10
        self.current.add_widget(label)
        self.push(label)
    elif cls is nodes.literal_block:
        box = RstLiteralBlock()
        self.current.add_widget(box)
        self.push(box)
    elif cls is nodes.emphasis:
        self.text += '[i]'
    elif cls is nodes.strong:
        self.text += '[b]'
    elif cls is nodes.literal:
        self.text += '[font=fonts/RobotoMono-Regular.ttf]'
    elif cls is nodes.block_quote:
        box = RstBlockQuote()
        self.current.add_widget(box)
        self.push(box.content)
        assert self.text == ''
    elif cls is nodes.enumerated_list:
        box = RstList()
        self.current.add_widget(box)
        self.push(box)
        self.idx_list = 0
    elif cls is nodes.bullet_list:
        box = RstList()
        self.current.add_widget(box)
        self.push(box)
        self.idx_list = None
    elif cls is nodes.list_item:
        bullet = '-'
        if self.idx_list is not None:
            self.idx_list += 1
            bullet = '%d.' % self.idx_list
        bullet = self.colorize(bullet, 'bullet')
        item = RstListItem()
        self.current.add_widget(RstListBullet(text=bullet, document=self.root))
        self.current.add_widget(item)
        self.push(item)
    elif cls is nodes.system_message:
        label = RstSystemMessage()
        if self.root.show_errors:
            self.current.add_widget(label)
        self.push(label)
    elif cls is nodes.warning:
        label = RstWarning()
        self.current.add_widget(label)
        self.push(label.content)
        assert self.text == ''
    elif cls is nodes.note:
        label = RstNote()
        self.current.add_widget(label)
        self.push(label.content)
        assert self.text == ''
    elif cls is nodes.image:
        uri = node['uri']
        align = node.get('align', 'center')
        image_size = [node.get('width'), node.get('height')]

        def set_size(img, size):
            img.size = [size[0] or img.width, size[1] or img.height]
        if uri.startswith('/') and self.root.document_root:
            uri = join(self.root.document_root, uri[1:])
        if uri.startswith('http://') or uri.startswith('https://'):
            image = RstAsyncImage(source=uri)
            image.bind(on_load=lambda *a: set_size(image, image_size))
        else:
            image = RstImage(source=uri)
            set_size(image, image_size)
        root = AnchorLayout(size_hint_y=None, anchor_x=align, height=image.height)
        image.bind(height=root.setter('height'))
        root.add_widget(image)
        self.current.add_widget(root)
    elif cls is nodes.definition_list:
        lst = RstDefinitionList(document=self.root)
        self.current.add_widget(lst)
        self.push(lst)
    elif cls is nodes.term:
        assert isinstance(self.current, RstDefinitionList)
        term = RstTerm(document=self.root)
        self.current.add_widget(term)
        self.push(term)
    elif cls is nodes.definition:
        assert isinstance(self.current, RstDefinitionList)
        definition = RstDefinition(document=self.root)
        definition.add_widget(RstDefinitionSpace(document=self.root))
        self.current.add_widget(definition)
        self.push(definition)
    elif cls is nodes.field_list:
        fieldlist = RstFieldList()
        self.current.add_widget(fieldlist)
        self.push(fieldlist)
    elif cls is nodes.field_name:
        name = RstFieldName(document=self.root)
        self.current.add_widget(name)
        self.push(name)
    elif cls is nodes.field_body:
        body = RstFieldBody()
        self.current.add_widget(body)
        self.push(body)
    elif cls is nodes.table:
        table = RstTable(cols=0)
        self.current.add_widget(table)
        self.push(table)
    elif cls is nodes.colspec:
        self.current.cols += 1
    elif cls is nodes.entry:
        entry = RstEntry()
        self.current.add_widget(entry)
        self.push(entry)
    elif cls is nodes.transition:
        self.current.add_widget(RstTransition())
    elif cls is nodes.reference:
        name = node.get('name', node.get('refuri'))
        self.text += '[ref=%s][color=%s]' % (name, self.root.colors.get('link', self.root.colors.get('paragraph')))
        if 'refname' in node and 'name' in node:
            self.root.refs_assoc[node['name']] = node['refname']
    elif cls is nodes.target:
        name = None
        if 'ids' in node:
            name = node['ids'][0]
        elif 'names' in node:
            name = node['names'][0]
        self.text += '[anchor=%s]' % name
        self.text_have_anchor = True
    elif cls is role_doc:
        self.doc_index = len(self.text)
    elif cls is role_video:
        pass