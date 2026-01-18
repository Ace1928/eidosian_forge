import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_draw_op(self, drawoperations, drawobj, stat, texlbl_name='texlbl', use_drawstring_pos=False):
    """Excecute the operations in drawoperations"""
    s = ''
    for drawop in drawoperations:
        op = drawop[0]
        style = getattr(drawobj, 'style', None)
        if style and (not self.options.get('duplicate')):
            style = self.filter_styles(style)
            styles = [self.styles.get(key.strip(), key.strip()) for key in style.split(',') if key]
            style = ','.join(styles)
        else:
            style = None
        if op in ['e', 'E']:
            s += self.draw_ellipse(drawop, style)
        elif op in ['p', 'P']:
            s += self.draw_polygon(drawop, style)
        elif op == 'L':
            s += self.draw_polyline(drawop, style)
        elif op in ['C', 'c']:
            s += self.set_color(drawop)
        elif op == 'S':
            s += self.set_style(drawop)
        elif op in ['B']:
            s += self.draw_bezier(drawop, style)
        elif op in ['T']:
            text = drawop[5]
            texmode = self.options.get('texmode', 'verbatim')
            if drawobj.attr.get('texmode', ''):
                texmode = drawobj.attr['texmode']
            if texlbl_name in drawobj.attr:
                text = drawobj.attr[texlbl_name]
            elif texmode == 'verbatim':
                text = escape_texchars(text)
                pass
            elif texmode == 'math':
                text = '$%s$' % text
            drawop[5] = text
            if self.options.get('alignstr', ''):
                drawop.append(self.options.get('alignstr'))
            if stat['T'] == 1 and self.options.get('valignmode', 'center') == 'center':
                drawop[3] = '0'
                if not use_drawstring_pos:
                    if texlbl_name == 'tailtexlbl':
                        pos = drawobj.attr.get('tail_lp') or drawobj.attr.get('pos')
                    elif texlbl_name == 'headtexlbl':
                        pos = drawobj.attr.get('head_lp') or drawobj.attr.get('pos')
                    else:
                        pos = drawobj.attr.get('lp') or drawobj.attr.get('pos')
                    if pos:
                        coord = pos.split(',')
                        if len(coord) == 2:
                            drawop[1] = coord[0]
                            drawop[2] = coord[1]
                        pass
            lblstyle = drawobj.attr.get('lblstyle')
            exstyle = drawobj.attr.get('exstyle', '')
            if exstyle:
                if lblstyle:
                    lblstyle += ',' + exstyle
                else:
                    lblstyle = exstyle
            s += self.draw_text(drawop, lblstyle)
    return s