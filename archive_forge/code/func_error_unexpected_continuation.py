def error_unexpected_continuation(self, line):
    raise self.parse_exc('Unexpected continuation line', self.lineno, line)