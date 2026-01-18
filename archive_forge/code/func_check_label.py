from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def check_label(label: Union[str, bytes, bytearray]) -> None:
    if isinstance(label, (bytes, bytearray)):
        label = label.decode('utf-8')
    if len(label) == 0:
        raise IDNAError('Empty Label')
    check_nfc(label)
    check_hyphen_ok(label)
    check_initial_combiner(label)
    for pos, cp in enumerate(label):
        cp_value = ord(cp)
        if intranges_contain(cp_value, idnadata.codepoint_classes['PVALID']):
            continue
        elif intranges_contain(cp_value, idnadata.codepoint_classes['CONTEXTJ']):
            try:
                if not valid_contextj(label, pos):
                    raise InvalidCodepointContext('Joiner {} not allowed at position {} in {}'.format(_unot(cp_value), pos + 1, repr(label)))
            except ValueError:
                raise IDNAError('Unknown codepoint adjacent to joiner {} at position {} in {}'.format(_unot(cp_value), pos + 1, repr(label)))
        elif intranges_contain(cp_value, idnadata.codepoint_classes['CONTEXTO']):
            if not valid_contexto(label, pos):
                raise InvalidCodepointContext('Codepoint {} not allowed at position {} in {}'.format(_unot(cp_value), pos + 1, repr(label)))
        else:
            raise InvalidCodepoint('Codepoint {} at position {} of {} not allowed'.format(_unot(cp_value), pos + 1, repr(label)))
    check_bidi(label)