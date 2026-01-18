import re
import numpy as np
import pytest
from mpl_toolkits.axisartist.angle_helper import (
@pytest.mark.parametrize('Formatter, regex', [(FormatterDMS, DMS_RE), (FormatterHMS, HMS_RE)], ids=['Degree/Minute/Second', 'Hour/Minute/Second'])
@pytest.mark.parametrize('direction, factor, values', [('left', 60, [0, -30, -60]), ('left', 600, [12301, 12302, 12303]), ('left', 3600, [0, -30, -60]), ('left', 36000, [738210, 738215, 738220]), ('left', 360000, [7382120, 7382125, 7382130]), ('left', 1.0, [45, 46, 47]), ('left', 10.0, [452, 453, 454])])
def test_formatters(Formatter, regex, direction, factor, values):
    fmt = Formatter()
    result = fmt(direction, factor, values)
    prev_degree = prev_minute = prev_second = None
    for tick, value in zip(result, values):
        m = regex.match(tick)
        assert m is not None, f'{tick!r} is not an expected tick format.'
        sign = sum((m.group(sign + '_sign') is not None for sign in ('degree', 'minute', 'second')))
        assert sign <= 1, f'Only one element of tick {tick!r} may have a sign.'
        sign = 1 if sign == 0 else -1
        degree = float(m.group('degree') or prev_degree or 0)
        minute = float(m.group('minute') or prev_minute or 0)
        second = float(m.group('second') or prev_second or 0)
        if Formatter == FormatterHMS:
            expected_value = pytest.approx(value // 15 / factor)
        else:
            expected_value = pytest.approx(value / factor)
        assert sign * dms2float(degree, minute, second) == expected_value, f'{tick!r} does not match expected tick value.'
        prev_degree = degree
        prev_minute = minute
        prev_second = second