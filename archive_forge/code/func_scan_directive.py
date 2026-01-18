from .error import MarkedYAMLError
from .tokens import *
def scan_directive(self):
    start_mark = self.get_mark()
    self.forward()
    name = self.scan_directive_name(start_mark)
    value = None
    if name == 'YAML':
        value = self.scan_yaml_directive_value(start_mark)
        end_mark = self.get_mark()
    elif name == 'TAG':
        value = self.scan_tag_directive_value(start_mark)
        end_mark = self.get_mark()
    else:
        end_mark = self.get_mark()
        while self.peek() not in '\x00\r\n\x85\u2028\u2029':
            self.forward()
    self.scan_directive_ignored_line(start_mark)
    return DirectiveToken(name, value, start_mark, end_mark)