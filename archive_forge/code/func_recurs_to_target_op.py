import sys
from itertools import groupby
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from ..._utils import logger_warning
from .. import LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS
from ._font import Font
from ._text_state_manager import TextStateManager
from ._text_state_params import TextStateParams
def recurs_to_target_op(ops: Iterator[Tuple[List[Any], bytes]], text_state_mgr: TextStateManager, end_target: Literal[b'Q', b'ET'], fonts: Dict[str, Font], strip_rotated: bool=True) -> Tuple[List[BTGroup], List[TextStateParams]]:
    """
    Recurse operators between BT/ET and/or q/Q operators managing the transform
    stack and capturing text positioning and rendering data.

    Args:
        ops: iterator of operators in content stream
        text_state_mgr: a TextStateManager instance
        end_target: Either b"Q" (ends b"q" op) or b"ET" (ends b"BT" op)
        fonts: font dictionary as returned by PageObject._layout_mode_fonts()

    Returns:
        tuple: list of BTGroup dicts + list of TextStateParams dataclass instances.
    """
    bt_groups: List[BTGroup] = []
    tj_ops: List[TextStateParams] = []
    if end_target == b'Q':
        text_state_mgr.add_q()
    while True:
        try:
            operands, op = next(ops)
        except StopIteration:
            return (bt_groups, tj_ops)
        if op == end_target:
            if op == b'Q':
                text_state_mgr.remove_q()
            if op == b'ET':
                if not tj_ops:
                    return (bt_groups, tj_ops)
                _text = ''
                bt_idx = 0
                last_displaced_tx = tj_ops[bt_idx].displaced_tx
                last_ty = tj_ops[bt_idx].ty
                for _idx, _tj in enumerate(tj_ops):
                    if strip_rotated and _tj.rotated:
                        continue
                    if abs(_tj.ty - last_ty) > _tj.font_height:
                        if _text.strip():
                            bt_groups.append(bt_group(tj_ops[bt_idx], _text, last_displaced_tx))
                        bt_idx = _idx
                        _text = ''
                    if last_displaced_tx - _tj.tx > _tj.space_tx * LAYOUT_NEW_BT_GROUP_SPACE_WIDTHS:
                        if _text.strip():
                            bt_groups.append(bt_group(tj_ops[bt_idx], _text, last_displaced_tx))
                        bt_idx = _idx
                        last_displaced_tx = _tj.displaced_tx
                        _text = ''
                    excess_tx = round(_tj.tx - last_displaced_tx, 3) * (_idx != bt_idx)
                    spaces = int(excess_tx // _tj.space_tx) if _tj.space_tx else 0
                    new_text = f'{' ' * spaces}{_tj.txt}'
                    last_ty = _tj.ty
                    _text = f'{_text}{new_text}'
                    last_displaced_tx = _tj.displaced_tx
                if _text:
                    bt_groups.append(bt_group(tj_ops[bt_idx], _text, last_displaced_tx))
                text_state_mgr.reset_tm()
            return (bt_groups, tj_ops)
        if op == b'q':
            bts, tjs = recurs_to_target_op(ops, text_state_mgr, b'Q', fonts, strip_rotated)
            bt_groups.extend(bts)
            tj_ops.extend(tjs)
        elif op == b'cm':
            text_state_mgr.add_cm(*operands)
        elif op == b'BT':
            bts, tjs = recurs_to_target_op(ops, text_state_mgr, b'ET', fonts, strip_rotated)
            bt_groups.extend(bts)
            tj_ops.extend(tjs)
        elif op == b'Tj':
            tj_ops.append(text_state_mgr.text_state_params(operands[0]))
        elif op == b'TJ':
            _tj = text_state_mgr.text_state_params()
            for tj_op in operands[0]:
                if isinstance(tj_op, bytes):
                    _tj = text_state_mgr.text_state_params(tj_op)
                    tj_ops.append(_tj)
                else:
                    text_state_mgr.add_trm(_tj.displacement_matrix(TD_offset=tj_op))
        elif op == b"'":
            text_state_mgr.reset_trm()
            text_state_mgr.add_tm([0, -text_state_mgr.TL])
            tj_ops.append(text_state_mgr.text_state_params(operands[0]))
        elif op == b'"':
            text_state_mgr.reset_trm()
            text_state_mgr.set_state_param(b'Tw', operands[0])
            text_state_mgr.set_state_param(b'Tc', operands[1])
            text_state_mgr.add_tm([0, -text_state_mgr.TL])
            tj_ops.append(text_state_mgr.text_state_params(operands[2]))
        elif op in (b'Td', b'Tm', b'TD', b'T*'):
            text_state_mgr.reset_trm()
            if op == b'Tm':
                text_state_mgr.reset_tm()
            elif op == b'TD':
                text_state_mgr.set_state_param(b'TL', -operands[1])
            elif op == b'T*':
                operands = [0, -text_state_mgr.TL]
            text_state_mgr.add_tm(operands)
        elif op == b'Tf':
            text_state_mgr.set_font(fonts[operands[0]], operands[1])
        else:
            text_state_mgr.set_state_param(op, operands)