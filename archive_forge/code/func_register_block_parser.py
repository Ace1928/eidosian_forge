import re
def register_block_parser(self, md, before=None):
    md.block.register(self.parser.name, self.directive_pattern, self.parse_directive, before=before)