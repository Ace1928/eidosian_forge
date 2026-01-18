from docutils import nodes, languages
from docutils.transforms import parts
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def backlinks(arg):
    value = directives.choice(arg, Contents.backlinks_values)
    if value == 'none':
        return None
    else:
        return value