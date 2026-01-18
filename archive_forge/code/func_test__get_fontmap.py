from pathlib import Path
import io
import pytest
from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
@pytest.mark.parametrize('family_name, file_name', [('WenQuanYi Zen Hei', 'wqy-zenhei'), ('Noto Sans CJK JP', 'NotoSansCJK')])
def test__get_fontmap(family_name, file_name):
    fp = fm.FontProperties(family=[family_name])
    found_file_name = Path(fm.findfont(fp)).name
    if file_name not in found_file_name:
        pytest.skip(f'Font {family_name} ({file_name}) is missing')
    text = 'There are 几个汉字 in between!'
    ft = fm.get_font(fm.fontManager._find_fonts_by_props(fm.FontProperties(family=['DejaVu Sans', family_name])))
    fontmap = ft._get_fontmap(text)
    for char, font in fontmap.items():
        if ord(char) > 127:
            assert Path(font.fname).name == found_file_name
        else:
            assert Path(font.fname).name == 'DejaVuSans.ttf'