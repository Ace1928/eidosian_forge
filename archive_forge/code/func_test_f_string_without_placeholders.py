from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_f_string_without_placeholders(self):
    self.flakes("f'foo'", m.FStringMissingPlaceholders)
    self.flakes('\n            f"""foo\n            bar\n            """\n        ', m.FStringMissingPlaceholders)
    self.flakes("\n            print(\n                f'foo'\n                f'bar'\n            )\n        ", m.FStringMissingPlaceholders)
    self.flakes("f'{{}}'", m.FStringMissingPlaceholders)
    self.flakes("\n            x = 5\n            print(f'{x}')\n        ")
    self.flakes("\n            x = 'a' * 90\n            print(f'{x:.8}')\n        ")
    self.flakes("\n            x = y = 5\n            print(f'{x:>2} {y:>2}')\n        ")