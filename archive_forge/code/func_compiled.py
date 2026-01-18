import re
from typing import Optional, Pattern, Match, Optional
@cached_property
def compiled(self) -> Pattern[str]:
    return re.compile(self.regex, self.flags)