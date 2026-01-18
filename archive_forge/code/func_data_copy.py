from typing import NamedTuple, Dict, Union, Iterator, Any
from emoji import unicode_codes
def data_copy(self) -> Dict[str, Any]:
    """
        Returns a copy of the data from :data:`EMOJI_DATA` for this match
        with the additional keys ``match_start`` and ``match_end``.
        """
    if self.data:
        emj_data = self.data.copy()
        emj_data['match_start'] = self.start
        emj_data['match_end'] = self.end
        return emj_data
    else:
        return {'match_start': self.start, 'match_end': self.end}