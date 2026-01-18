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
def brute_refs(self, node):

    def get_refs(condition, backref=False):
        autonum = autosym = 0
        _nodes = node.traverse(condition=condition, ascend=False)
        for f in _nodes:
            id = f['ids'][0]
            auto = ''
            if 'auto' in f:
                auto = f['auto']
            if auto == 1:
                autonum += 1
                key = 'backref' + str(autonum) if backref else str(autonum)
                self.root.refs_assoc[key] = id
            elif auto == '*':
                sym = self.footlist[autosym % 10] * (int(autosym / 10) + 1)
                key = 'backref' + sym if backref else sym
                self.root.refs_assoc[key] = id
                autosym += 1
            else:
                if not backref:
                    key = f['names'][0]
                    if key:
                        self.root.refs_assoc[key] = id
                    continue
                key = 'backref' + f['refname'][0]
                if key in self.root.refs_assoc:
                    self.root.refs_assoc[key].append(id)
                else:
                    self.root.refs_assoc[key] = [id]
    get_refs(nodes.footnote, backref=False)
    get_refs(nodes.footnote_reference, backref=True)