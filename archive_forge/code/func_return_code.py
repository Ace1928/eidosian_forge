from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from click import style
from black.output import err, out
@property
def return_code(self) -> int:
    """Return the exit code that the app should use.

        This considers the current state of changed files and failures:
        - if there were any failures, return 123;
        - if any files were changed and --check is being used, return 1;
        - otherwise return 0.
        """
    if self.failure_count:
        return 123
    elif self.change_count and self.check:
        return 1
    return 0