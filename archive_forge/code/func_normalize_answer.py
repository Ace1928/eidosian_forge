import re
from typing import Optional
def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search('^\\\\text\\{(?P<text>.+?)\\}$', answer)
        if m is not None:
            answer = m.group('text').strip()
        return _strip_string(answer)
    except:
        return answer