from __future__ import annotations
import abc
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Type, TYPE_CHECKING
@property
def aj2(self) -> 'jinja2.Environment':
    """
        Returns the jinja2 context
        """
    if self._aj2 is None:
        self._aj2 = self.settings.ctx.get_j2_ctx(path=self.base_path or 'assets/sql', name=self.name, enable_async=True, comment_start_string='/*', comment_end_string='*/')
    return self._aj2