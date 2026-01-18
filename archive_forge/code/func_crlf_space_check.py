import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..generic import DictionaryObject, TextStringObject, encode_pdfdocencoding
def crlf_space_check(text: str, cmtm_prev: Tuple[List[float], List[float]], cmtm_matrix: Tuple[List[float], List[float]], memo_cmtm: Tuple[List[float], List[float]], cmap: Tuple[Union[str, Dict[int, str]], Dict[str, str], str, Optional[DictionaryObject]], orientations: Tuple[int, ...], output: str, font_size: float, visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]], spacewidth: float) -> Tuple[str, str, List[float], List[float]]:
    cm_prev = cmtm_prev[0]
    tm_prev = cmtm_prev[1]
    cm_matrix = cmtm_matrix[0]
    tm_matrix = cmtm_matrix[1]
    memo_cm = memo_cmtm[0]
    memo_tm = memo_cmtm[1]
    m_prev = mult(tm_prev, cm_prev)
    m = mult(tm_matrix, cm_matrix)
    orientation = orient(m)
    delta_x = m[4] - m_prev[4]
    delta_y = m[5] - m_prev[5]
    k = math.sqrt(abs(m[0] * m[3]) + abs(m[1] * m[2]))
    f = font_size * k
    cm_prev = m
    if orientation not in orientations:
        raise OrientationNotFoundError
    try:
        if orientation == 0:
            if delta_y < -0.8 * f:
                if (output + text)[-1] != '\n':
                    output += text + '\n'
                    if visitor_text is not None:
                        visitor_text(text + '\n', memo_cm, memo_tm, cmap[3], font_size)
                    text = ''
            elif abs(delta_y) < f * 0.3 and abs(delta_x) > spacewidth * f * 15 and ((output + text)[-1] != ' '):
                text += ' '
        elif orientation == 180:
            if delta_y > 0.8 * f:
                if (output + text)[-1] != '\n':
                    output += text + '\n'
                    if visitor_text is not None:
                        visitor_text(text + '\n', memo_cm, memo_tm, cmap[3], font_size)
                    text = ''
            elif abs(delta_y) < f * 0.3 and abs(delta_x) > spacewidth * f * 15 and ((output + text)[-1] != ' '):
                text += ' '
        elif orientation == 90:
            if delta_x > 0.8 * f:
                if (output + text)[-1] != '\n':
                    output += text + '\n'
                    if visitor_text is not None:
                        visitor_text(text + '\n', memo_cm, memo_tm, cmap[3], font_size)
                    text = ''
            elif abs(delta_x) < f * 0.3 and abs(delta_y) > spacewidth * f * 15 and ((output + text)[-1] != ' '):
                text += ' '
        elif orientation == 270:
            if delta_x < -0.8 * f:
                if (output + text)[-1] != '\n':
                    output += text + '\n'
                    if visitor_text is not None:
                        visitor_text(text + '\n', memo_cm, memo_tm, cmap[3], font_size)
                    text = ''
            elif abs(delta_x) < f * 0.3 and abs(delta_y) > spacewidth * f * 15 and ((output + text)[-1] != ' '):
                text += ' '
    except Exception:
        pass
    tm_prev = tm_matrix.copy()
    cm_prev = cm_matrix.copy()
    return (text, output, cm_prev, tm_prev)