def matching(self, pattern):
    result = self._create()
    result._match_pattern = self._input.get_regexp(pattern)
    result._update()
    return result