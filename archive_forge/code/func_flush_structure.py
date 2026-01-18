import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def flush_structure(self, docs_link=None):
    """Flushes a doc structure to a ReSTructed string

        The document is flushed out in a DFS style where sections and their
        subsections' values are added to the string as they are visited.
        """
    path_length = len(self.path)
    if path_length == 1:
        if self.hrefs:
            self.style.new_paragraph()
            for refname, link in self.hrefs.items():
                self.style.link_target_definition(refname, link)
    elif path_length == SECTION_METHOD_PATH_DEPTH.get(self.path[1]):
        docs_link = None
    value = self.getvalue()
    for name, section in self._structure.items():
        match = DOCUMENTATION_LINK_REGEX.search(value.decode())
        docs_link = f'{match.group(0)}\n\n'.encode() if match else docs_link
        value += section.flush_structure(docs_link)
    line_count = len(value.splitlines())
    section_config = SECTION_LINE_LIMIT_CONFIG.get(self.name)
    aws_docs_link = docs_link.decode() if docs_link is not None else DEFAULT_AWS_DOCS_LINK
    if section_config and line_count > section_config['line_limit']:
        value = LARGE_SECTION_MESSAGE.format(section_config['name'], aws_docs_link).encode()
    return value