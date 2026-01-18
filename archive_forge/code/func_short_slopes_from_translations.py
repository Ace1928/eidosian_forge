from ..sage_helper import _within_sage
import math
def short_slopes_from_translations(translations, length=6):
    m_tran, l_tran = translations
    if _within_sage:
        if is_ComplexIntervalFieldElement(m_tran):
            raise Exception('Meridian translation expected to be real')
        if is_RealIntervalFieldElement(l_tran):
            raise Exception('Longitude translation expected to be complex')
        is_interval_1 = is_RealIntervalFieldElement(m_tran)
        is_interval_2 = is_ComplexIntervalFieldElement(l_tran)
        if is_interval_1 != is_interval_2:
            raise Exception('Mismatch of non-intervals and intervals.')
        if is_interval_1:
            return _verified_short_slopes_from_translations(translations, length)
    return _unverified_short_slopes_from_translations(translations, length)