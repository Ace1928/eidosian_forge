def error_empty_key(self, line):
    raise self.parse_exc('Key cannot be empty', self.lineno, line)