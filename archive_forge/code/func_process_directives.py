from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def process_directives(self):
    self.yaml_version = None
    self.tag_handles = {}
    while self.check_token(DirectiveToken):
        token = self.get_token()
        if token.name == 'YAML':
            if self.yaml_version is not None:
                raise ParserError(None, None, 'found duplicate YAML directive', token.start_mark)
            major, minor = token.value
            if major != 1:
                raise ParserError(None, None, 'found incompatible YAML document (version 1.* is required)', token.start_mark)
            self.yaml_version = token.value
        elif token.name == 'TAG':
            handle, prefix = token.value
            if handle in self.tag_handles:
                raise ParserError(None, None, 'duplicate tag handle %r' % handle, token.start_mark)
            self.tag_handles[handle] = prefix
    if self.tag_handles:
        value = (self.yaml_version, self.tag_handles.copy())
    else:
        value = (self.yaml_version, None)
    for key in self.DEFAULT_TAGS:
        if key not in self.tag_handles:
            self.tag_handles[key] = self.DEFAULT_TAGS[key]
    return value