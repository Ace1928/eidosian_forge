import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
def fonts_to_quadratic(fonts, max_err_em=None, max_err=None, reverse_direction=False, stats=None, dump_stats=False, remember_curve_type=True, all_quadratic=True):
    """Convert the curves of a collection of fonts to quadratic.

    All curves will be converted to quadratic at once, ensuring interpolation
    compatibility. If this is not required, calling fonts_to_quadratic with one
    font at a time may yield slightly more optimized results.

    Return the set of modified glyph names if any, else return an empty set.

    By default, cu2qu stores the curve type in the fonts' lib, under a private
    key "com.github.googlei18n.cu2qu.curve_type", and will not try to convert
    them again if the curve type is already set to "quadratic".
    Setting 'remember_curve_type' to False disables this optimization.

    Raises IncompatibleFontsError if same-named glyphs from different fonts
    have non-interpolatable outlines.
    """
    if remember_curve_type:
        curve_types = {f.lib.get(CURVE_TYPE_LIB_KEY, 'cubic') for f in fonts}
        if len(curve_types) == 1:
            curve_type = next(iter(curve_types))
            if curve_type in ('quadratic', 'mixed'):
                logger.info('Curves already converted to quadratic')
                return False
            elif curve_type == 'cubic':
                pass
            else:
                raise NotImplementedError(curve_type)
        elif len(curve_types) > 1:
            logger.warning('fonts may contain different curve types')
    if stats is None:
        stats = {}
    if max_err_em and max_err:
        raise TypeError('Only one of max_err and max_err_em can be specified.')
    if not (max_err_em or max_err):
        max_err_em = DEFAULT_MAX_ERR
    if isinstance(max_err, (list, tuple)):
        assert len(max_err) == len(fonts)
        max_errors = max_err
    elif max_err:
        max_errors = [max_err] * len(fonts)
    if isinstance(max_err_em, (list, tuple)):
        assert len(fonts) == len(max_err_em)
        max_errors = [f.info.unitsPerEm * e for f, e in zip(fonts, max_err_em)]
    elif max_err_em:
        max_errors = [f.info.unitsPerEm * max_err_em for f in fonts]
    modified = set()
    glyph_errors = {}
    for name in set().union(*(f.keys() for f in fonts)):
        glyphs = []
        cur_max_errors = []
        for font, error in zip(fonts, max_errors):
            if name in font:
                glyphs.append(font[name])
                cur_max_errors.append(error)
        try:
            if _glyphs_to_quadratic(glyphs, cur_max_errors, reverse_direction, stats, all_quadratic):
                modified.add(name)
        except IncompatibleGlyphsError as exc:
            logger.error(exc)
            glyph_errors[name] = exc
    if glyph_errors:
        raise IncompatibleFontsError(glyph_errors)
    if modified and dump_stats:
        spline_lengths = sorted(stats.keys())
        logger.info('New spline lengths: %s' % ', '.join(('%s: %d' % (l, stats[l]) for l in spline_lengths)))
    if remember_curve_type:
        for font in fonts:
            curve_type = font.lib.get(CURVE_TYPE_LIB_KEY, 'cubic')
            new_curve_type = 'quadratic' if all_quadratic else 'mixed'
            if curve_type != new_curve_type:
                font.lib[CURVE_TYPE_LIB_KEY] = new_curve_type
    return modified