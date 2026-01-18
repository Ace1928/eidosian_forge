import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata
from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning
def pretty_symbol(symb_name, bold_name=False):
    """return pretty representation of a symbol"""
    if not _use_unicode:
        return symb_name
    name, sups, subs = split_super_sub(symb_name)

    def translate(s, bold_name):
        if bold_name:
            gG = greek_bold_unicode.get(s)
        else:
            gG = greek_unicode.get(s)
        if gG is not None:
            return gG
        for key in sorted(modifier_dict.keys(), key=lambda k: len(k), reverse=True):
            if s.lower().endswith(key) and len(s) > len(key):
                return modifier_dict[key](translate(s[:-len(key)], bold_name))
        if bold_name:
            return ''.join([bold_unicode[c] for c in s])
        return s
    name = translate(name, bold_name)

    def pretty_list(l, mapping):
        result = []
        for s in l:
            pretty = mapping.get(s)
            if pretty is None:
                try:
                    pretty = ''.join([mapping[c] for c in s])
                except (TypeError, KeyError):
                    return None
            result.append(pretty)
        return result
    pretty_sups = pretty_list(sups, sup)
    if pretty_sups is not None:
        pretty_subs = pretty_list(subs, sub)
    else:
        pretty_subs = None
    if pretty_subs is None:
        if subs:
            name += '_' + '_'.join([translate(s, bold_name) for s in subs])
        if sups:
            name += '__' + '__'.join([translate(s, bold_name) for s in sups])
        return name
    else:
        sups_result = ' '.join(pretty_sups)
        subs_result = ' '.join(pretty_subs)
    return ''.join([name, sups_result, subs_result])