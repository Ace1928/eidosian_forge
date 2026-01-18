from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_nonRecurseImportErrors(self) -> str:
    d = self.runTrial('-N', 'package2')
    self.assertIn(d, '[ERROR]')
    self.assertIn(d, _noModuleError)
    self.assertNotIn(d, '<module ')
    return d