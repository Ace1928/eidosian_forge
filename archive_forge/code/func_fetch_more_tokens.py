from .error import MarkedYAMLError
from .tokens import *
def fetch_more_tokens(self):
    self.scan_to_next_token()
    self.stale_possible_simple_keys()
    self.unwind_indent(self.column)
    ch = self.peek()
    if ch == '\x00':
        return self.fetch_stream_end()
    if ch == '%' and self.check_directive():
        return self.fetch_directive()
    if ch == '-' and self.check_document_start():
        return self.fetch_document_start()
    if ch == '.' and self.check_document_end():
        return self.fetch_document_end()
    if ch == '[':
        return self.fetch_flow_sequence_start()
    if ch == '{':
        return self.fetch_flow_mapping_start()
    if ch == ']':
        return self.fetch_flow_sequence_end()
    if ch == '}':
        return self.fetch_flow_mapping_end()
    if ch == ',':
        return self.fetch_flow_entry()
    if ch == '-' and self.check_block_entry():
        return self.fetch_block_entry()
    if ch == '?' and self.check_key():
        return self.fetch_key()
    if ch == ':' and self.check_value():
        return self.fetch_value()
    if ch == '*':
        return self.fetch_alias()
    if ch == '&':
        return self.fetch_anchor()
    if ch == '!':
        return self.fetch_tag()
    if ch == '|' and (not self.flow_level):
        return self.fetch_literal()
    if ch == '>' and (not self.flow_level):
        return self.fetch_folded()
    if ch == "'":
        return self.fetch_single()
    if ch == '"':
        return self.fetch_double()
    if self.check_plain():
        return self.fetch_plain()
    raise ScannerError('while scanning for the next token', None, 'found character %r that cannot start any token' % ch, self.get_mark())