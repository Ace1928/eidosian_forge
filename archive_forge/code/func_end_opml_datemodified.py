from __future__ import annotations
import copy
from . import common, dates
def end_opml_datemodified(self) -> None:
    value = self.get_text()
    if value:
        self.harvest['meta']['modified'] = value
        timestamp = dates.parse_rfc822(value)
        if timestamp:
            self.harvest['meta']['modified_parsed'] = timestamp
        else:
            self.raise_bozo('dateModified is not an RFC 822 datetime')