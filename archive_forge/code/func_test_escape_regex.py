import yaql.tests
def test_escape_regex(self):
    self.assertEqual('\\[', self.eval("escapeRegex('[')"))