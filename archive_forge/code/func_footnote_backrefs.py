import os.path
import docutils
from docutils import frontend, nodes, writers, io
from docutils.transforms import writer_aux
from docutils.writers import _html_base
def footnote_backrefs(self, node):
    backlinks = []
    backrefs = node['backrefs']
    if self.settings.footnote_backlinks and backrefs:
        if len(backrefs) == 1:
            self.context.append('')
            self.context.append('</a>')
            self.context.append('<a class="fn-backref" href="#%s">' % backrefs[0])
        else:
            for i, backref in enumerate(backrefs, 1):
                backlinks.append('<a class="fn-backref" href="#%s">%s</a>' % (backref, i))
            self.context.append('<em>(%s)</em> ' % ', '.join(backlinks))
            self.context += ['', '']
    else:
        self.context.append('')
        self.context += ['', '']
    if len(node) > 1:
        if not backlinks:
            node[1]['classes'].append('first')
        node[-1]['classes'].append('last')