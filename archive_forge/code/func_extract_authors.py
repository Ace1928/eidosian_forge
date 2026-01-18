import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def extract_authors(self, field, name, docinfo):
    try:
        if len(field[1]) == 1:
            if isinstance(field[1][0], nodes.paragraph):
                authors = self.authors_from_one_paragraph(field)
            elif isinstance(field[1][0], nodes.bullet_list):
                authors = self.authors_from_bullet_list(field)
            else:
                raise TransformError
        else:
            authors = self.authors_from_paragraphs(field)
        authornodes = [nodes.author('', '', *author) for author in authors if author]
        if len(authornodes) >= 1:
            docinfo.append(nodes.authors('', *authornodes))
        else:
            raise TransformError
    except TransformError:
        field[-1] += self.document.reporter.warning('Bibliographic field "%s" incompatible with extraction: it must contain either a single paragraph (with authors separated by one of "%s"), multiple paragraphs (one per author), or a bullet list with one paragraph (one author) per item.' % (name, ''.join(self.language.author_separators)), base_node=field)
        raise