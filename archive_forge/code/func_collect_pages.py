import posixpath
import traceback
from os import path
from typing import Any, Dict, Generator, Iterable, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import get_full_modname, logging, status_iterator
from sphinx.util.nodes import make_refnode
def collect_pages(app: Sphinx) -> Generator[Tuple[str, Dict[str, Any], str], None, None]:
    env = app.builder.env
    if not hasattr(env, '_viewcode_modules'):
        return
    if not is_supported_builder(app.builder):
        return
    highlighter = app.builder.highlighter
    urito = app.builder.get_relative_uri
    modnames = set(env._viewcode_modules)
    for modname, entry in status_iterator(sorted(env._viewcode_modules.items()), __('highlighting module code... '), 'blue', len(env._viewcode_modules), app.verbosity, lambda x: x[0]):
        if not entry:
            continue
        if not should_generate_module_page(app, modname):
            continue
        code, tags, used, refname = entry
        pagename = posixpath.join(OUTPUT_DIRNAME, modname.replace('.', '/'))
        if env.config.highlight_language in {'default', 'none'}:
            lexer = env.config.highlight_language
        else:
            lexer = 'python'
        highlighted = highlighter.highlight_block(code, lexer, linenos=False)
        lines = highlighted.splitlines()
        before, after = lines[0].split('<pre>')
        lines[0:1] = [before + '<pre>', after]
        maxindex = len(lines) - 1
        for name, docname in used.items():
            type, start, end = tags[name]
            backlink = urito(pagename, docname) + '#' + refname + '.' + name
            lines[start] = '<div class="viewcode-block" id="%s"><a class="viewcode-back" href="%s">%s</a>' % (name, backlink, _('[docs]')) + lines[start]
            lines[min(end, maxindex)] += '</div>'
        parents = []
        parent = modname
        while '.' in parent:
            parent = parent.rsplit('.', 1)[0]
            if parent in modnames:
                parents.append({'link': urito(pagename, posixpath.join(OUTPUT_DIRNAME, parent.replace('.', '/'))), 'title': parent})
        parents.append({'link': urito(pagename, posixpath.join(OUTPUT_DIRNAME, 'index')), 'title': _('Module code')})
        parents.reverse()
        context = {'parents': parents, 'title': modname, 'body': _('<h1>Source code for %s</h1>') % modname + '\n'.join(lines)}
        yield (pagename, context, 'page.html')
    if not modnames:
        return
    html = ['\n']
    stack = ['']
    for modname in sorted(modnames):
        if modname.startswith(stack[-1]):
            stack.append(modname + '.')
            html.append('<ul>')
        else:
            stack.pop()
            while not modname.startswith(stack[-1]):
                stack.pop()
                html.append('</ul>')
            stack.append(modname + '.')
        html.append('<li><a href="%s">%s</a></li>\n' % (urito(posixpath.join(OUTPUT_DIRNAME, 'index'), posixpath.join(OUTPUT_DIRNAME, modname.replace('.', '/'))), modname))
    html.append('</ul>' * (len(stack) - 1))
    context = {'title': _('Overview: module code'), 'body': _('<h1>All modules for which code is available</h1>') + ''.join(html)}
    yield (posixpath.join(OUTPUT_DIRNAME, 'index'), context, 'page.html')