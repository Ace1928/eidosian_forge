import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def check_empty_biblio_field(self, field, name):
    if len(field[-1]) < 1:
        field[-1] += self.document.reporter.warning('Cannot extract empty bibliographic field "%s".' % name, base_node=field)
        return None
    return 1