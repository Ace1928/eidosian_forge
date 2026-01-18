import re
from typing import Dict, Match
def evalString(s: str) -> str:
    assert s.startswith("'") or s.startswith('"'), repr(s[:1])
    q = s[0]
    if s[:3] == q * 3:
        q = q * 3
    assert s.endswith(q), repr(s[-len(q):])
    assert len(s) >= 2 * len(q)
    s = s[len(q):-len(q)]
    return re.sub('\\\\(\\\'|\\"|\\\\|[abfnrtv]|x.{0,2}|[0-7]{1,3})', escape, s)