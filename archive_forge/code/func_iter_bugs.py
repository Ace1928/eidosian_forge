from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def iter_bugs(self):
    """Iterate over the bugs associated with this revision."""
    bug_property = self.properties.get('bugs', None)
    if bug_property is None:
        return iter([])
    from . import bugtracker
    return bugtracker.decode_bug_urls(bug_property)