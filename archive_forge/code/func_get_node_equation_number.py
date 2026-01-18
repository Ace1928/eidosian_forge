from typing import Optional
from docutils import nodes
from sphinx.builders.html import HTMLTranslator
def get_node_equation_number(writer: HTMLTranslator, node: nodes.math_block) -> str:
    if writer.builder.config.math_numfig and writer.builder.config.numfig:
        figtype = 'displaymath'
        if writer.builder.name == 'singlehtml':
            key = '%s/%s' % (writer.docnames[-1], figtype)
        else:
            key = figtype
        id = node['ids'][0]
        number = writer.builder.fignumbers.get(key, {}).get(id, ())
        return '.'.join(map(str, number))
    else:
        return node['number']