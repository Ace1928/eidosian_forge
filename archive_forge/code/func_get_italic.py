from pygame._freetype import init, Font as _Font, get_default_resolution
from pygame._freetype import quit, get_default_font, get_init as _get_init
from pygame._freetype import _internal_mod_init
from pygame.sysfont import match_font, get_fonts, SysFont as _SysFont
from pygame import encode_file_path
def get_italic(self):
    """get_italic() -> bool
        check if the text will be rendered italic"""
    return self.oblique