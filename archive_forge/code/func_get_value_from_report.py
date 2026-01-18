from enum import Enum
from typing import Dict, Optional
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def get_value_from_report(self, report: Dict) -> Optional[str]:
    """Returns `None` if the tag isn't in the report."""
    if 'extra_usage_tags' not in report:
        return None
    return report['extra_usage_tags'].get(TagKey.Name(self.value).lower(), None)