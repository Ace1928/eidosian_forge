from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def parse_actor_and_date(line: str) -> Tuple[Actor, int, int]:
    """Parse out the actor (author or committer) info from a line like::

        author Tom Preston-Werner <tom@mojombo.com> 1191999972 -0700

    :return: [Actor, int_seconds_since_epoch, int_timezone_offset]
    """
    actor, epoch, offset = ('', '0', '0')
    m = _re_actor_epoch.search(line)
    if m:
        actor, epoch, offset = m.groups()
    else:
        m = _re_only_actor.search(line)
        actor = m.group(1) if m else line or ''
    return (Actor._from_string(actor), int(epoch), utctz_to_altz(offset))