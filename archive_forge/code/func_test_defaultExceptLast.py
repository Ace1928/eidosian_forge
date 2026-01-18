from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_defaultExceptLast(self):
    """
        A default except block should be last.

        YES:

        try:
            ...
        except Exception:
            ...
        except:
            ...

        NO:

        try:
            ...
        except:
            ...
        except Exception:
            ...
        """
    self.flakes('\n        try:\n            pass\n        except ValueError:\n            pass\n        ')
    self.flakes('\n        try:\n            pass\n        except ValueError:\n            pass\n        except:\n            pass\n        ')
    self.flakes('\n        try:\n            pass\n        except:\n            pass\n        ')
    self.flakes('\n        try:\n            pass\n        except ValueError:\n            pass\n        else:\n            pass\n        ')
    self.flakes('\n        try:\n            pass\n        except:\n            pass\n        else:\n            pass\n        ')
    self.flakes('\n        try:\n            pass\n        except ValueError:\n            pass\n        except:\n            pass\n        else:\n            pass\n        ')