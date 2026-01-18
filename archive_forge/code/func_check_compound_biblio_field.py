import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def check_compound_biblio_field(self, field, name):
    if len(field[-1]) > 1:
        field[-1] += self.document.reporter.warning('Cannot extract compound bibliographic field "%s".' % name, base_node=field)
        return None
    if not isinstance(field[-1][0], nodes.paragraph):
        field[-1] += self.document.reporter.warning('Cannot extract bibliographic field "%s" containing anything other than a single paragraph.' % name, base_node=field)
        return None
    return 1