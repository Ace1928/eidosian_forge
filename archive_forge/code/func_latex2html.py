import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None
def latex2html(node, source):
    inline = isinstance(node.parent, nodes.TextElement)
    latex = node['latex']
    fontset = node['fontset']
    fontsize = node['fontsize']
    name = 'math-{}'.format(hashlib.md5(f'{latex}{fontset}{fontsize}'.encode()).hexdigest()[-10:])
    destdir = Path(setup.app.builder.outdir, '_images', 'mathmpl')
    destdir.mkdir(parents=True, exist_ok=True)
    dest = destdir / f'{name}.png'
    depth = latex2png(latex, dest, fontset, fontsize=fontsize)
    srcset = []
    for size in setup.app.config.mathmpl_srcset:
        filename = f'{name}-{size.replace('.', '_')}.png'
        latex2png(latex, destdir / filename, fontset, fontsize=fontsize, dpi=100 * float(size[:-1]))
        srcset.append(f'{setup.app.builder.imgpath}/mathmpl/{filename} {size}')
    if srcset:
        srcset = f'srcset="{setup.app.builder.imgpath}/mathmpl/{name}.png, ' + ', '.join(srcset) + '" '
    if inline:
        cls = ''
    else:
        cls = 'class="center" '
    if inline and depth != 0:
        style = 'style="position: relative; bottom: -%dpx"' % (depth + 1)
    else:
        style = ''
    return f'<img src="{setup.app.builder.imgpath}/mathmpl/{name}.png" {srcset}{cls}{style}/>'