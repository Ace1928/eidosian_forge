import os
import re
def restrict_rglob(self, pattern):
    """
        Raise ValueError if ** appears in anything but a full path segment.

        >>> Translator().translate('**foo')
        Traceback (most recent call last):
        ...
        ValueError: ** must appear alone in a path segment
        """
    seps_pattern = f'[{re.escape(self.seps)}]+'
    segments = re.split(seps_pattern, pattern)
    if any(('**' in segment and segment != '**' for segment in segments)):
        raise ValueError('** must appear alone in a path segment')