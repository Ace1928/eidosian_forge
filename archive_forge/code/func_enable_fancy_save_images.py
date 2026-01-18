from .gui import *
from . import smooth
from .colors import Palette
from .manager import LinkManager
def enable_fancy_save_images(self):
    fancy = [3, 4] if have_pyx else [3]
    for i in fancy:
        self.save_image_menu.entryconfig(i, state='active')