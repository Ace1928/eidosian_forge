import datetime
import re
import sys
import time
from ..repo import Repo
def get_current_version(projdir=PROJDIR, pattern=PATTERN, logger=None):
    """Return the most recent tag, using an options regular expression pattern.

    The default pattern will strip any characters preceding the first semantic
    version. *EG*: "Release-0.2.1-rc.1" will be come "0.2.1-rc.1". If no match
    is found, then the most recent tag is return without modification.

    Args:
      projdir: path to ``.git``
      pattern: regular expression pattern with group that matches version
      logger: a Python logging instance to capture exception
    Returns:
      tag matching first group in regular expression pattern
    """
    tags = get_recent_tags(projdir)
    try:
        tag = tags[0][0]
    except IndexError:
        return
    matches = re.match(pattern, tag)
    try:
        current_version = matches.group(1)
    except (IndexError, AttributeError) as err:
        if logger:
            logger.exception(err)
        return tag
    return current_version